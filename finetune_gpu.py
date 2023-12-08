#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

"""
python3 finetune_gpu.py --restore_from trial6_gpu --extract_ablated_feat --val_batch_size 64 --layer 9
"""

import argparse
import json
import os, sys
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from tensorflow.python.training.checkpoint_utils import load_checkpoint
import time
from pathlib import Path
import random
from functools import partial
from tqdm import tqdm
from collections import defaultdict

user_home_dir = str(Path.home())
print(user_home_dir)
not_user_home_dir = "/mnt/data/angelaz2"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

if tf.VERSION >= "2":
	tf.disable_eager_execution()
	tf.config.experimental.enable_tensor_float_32_execution(False)
	tf.config.optimizer.set_experimental_options({"layout_optimizer": False,
												  "constant_folding": False,
												  "shape_optimization": False,
												  "remapping": False,
												  "arithmetic_optimization": False,
												  "dependency_optimization": False,
												  "loop_optimization": False,
												  "disable_meta_optimizer": True
												  })

from load_story_dataset import load_dataset, load_eval_dataset
import model as gpt2_model

sys.path.insert(0, os.path.join(user_home_dir, "transformers/gpt-2_ft"))
#import src.encoder as encoder
from src.load_dataset import Sampler
from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from utils_model import find_trainable_variables, average_grads, assign_to_gpu
from phrase_level_model_config import *

opt_fns = {
	'adam':adam,
}

lr_schedules = {
	'warmup_cosine':warmup_cosine,
	'warmup_linear':warmup_linear,
	'warmup_constant':warmup_constant,
}

def maketree(path):
	try:
		os.makedirs(path)
	except:
		pass

def save():
	print(
		"Saving",
		os.path.join(CHECKPOINT_DIR, args.run_name,
					 "model-{}").format(counter))
	saver.save(
		sess,
		os.path.join(CHECKPOINT_DIR, args.run_name, "model"),
		global_step=counter)
	with open(counter_path, "w") as fp:
		fp.write(str(counter) + "\n")

def sample_batch():
	return [data_sampler.sample(hparams.n_ctx) for _ in range(n_batch_train)]

def sample_eval_chunks(chunks, n_ctx):
	split_chunks = []
	for story_chunk in chunks:
		for i in range(0, story_chunk.shape[0], n_ctx):
			if i+n_ctx >= story_chunk.shape[0]: continue
			split_chunks.append(story_chunk[i:i+n_ctx])
	split_chunks = np.array(split_chunks)
	return split_chunks

def sample_cts_eval_chunks(chunks, n_ctx, unk_id):
	split_chunks = {}
	for story in chunks:
		nword = chunks[story].shape[0]
		cts_chunks = unk_id * np.ones([nword, n_ctx], dtype=np.int32)
		for i in range(0, n_ctx):
			cts_chunks[:nword-i, i] = chunks[story][i:]
		split_chunks[story] = cts_chunks
	return split_chunks

def mgpu_train(xs, hparams):
	gpu_ops = []
	gpu_grads = []
	xs = tf.split(xs, args.n_gpu, 0)
	for i, xs in enumerate(xs):
		do_reuse = True if i > 0 else None
		with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
			train_output = gpt2_model.model(hparams=hparams, X=xs, reuse=do_reuse)
			train_loss = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=xs[:, 1:], logits=train_output["logits"][:, :-1]))
			params = find_trainable_variables("model")
			grads = tf.gradients(train_loss, params)
			grads = list(zip(grads, params))
			gpu_grads.append(grads)
			gpu_ops.append([train_loss])
	ops = [tf.concat(op, 0) for op in gpu_ops]
	grads = average_grads(gpu_grads)
	grads = [g for g, p in grads]
	train = opt_fns[args.optimizer](
		params, grads, args.lr, partial(lr_schedules[args.lr_schedule], warmup=args.lr_warmup),
		n_updates_total, l2=args.l2, max_grad_norm=args.max_grad_norm, vector_l2=args.vector_l2, 
		b1=args.b1, b2=args.b2, e=args.e)
	return train, ops

def mgpu_predict(xs, hparams):
	xs = tf.split(xs, args.n_gpu, 0)
	v_val_loss, v_val_ppx, v_val_states = [], [], []
	for i, xs in enumerate(xs):
		with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
			val_output = gpt2_model.model(hparams=hparams, X=xs, reuse=True)
			val_loss = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=xs[:, 1:], logits=val_output["logits"][:, :-1]))
			v_val_loss.append(val_loss)
			v_val_ppx.append(val_output["ppx"])
			v_val_states.append(val_output["states"])
	return v_val_loss, v_val_ppx, tf.concat(v_val_states, 0)

def mgpu_predict_single(xs, hparams):
	with tf.device(assign_to_gpu(0, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=True):
		val_output = gpt2_model.model(hparams=hparams, X=xs, reuse=True)
		v_val_loss = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=xs[:, 1:], logits=val_output["logits"][:, :-1]))
		v_val_ppx = val_output["ppx"]
		v_val_states = val_output["states"]
	return v_val_states

def get_ablated_hs(ablated_chunk, pos, layer, cl):
	if pos <= cl:
		ablated_chunk[:, pos] = unk_id
	nwords = ablated_chunk.shape[0]
	nbatch = args.val_batch_size * args.n_gpu
	hs = []
	for i in range(0, nwords, nbatch):
		if i+nbatch >= nwords:
			for j in range(i, nwords, args.n_gpu):
				if j+args.n_gpu >= nwords:
					for k in range(j, nwords):
						h = sess.run(val_single_states,
									 feed_dict={val_single_context: [ablated_chunk[k]]})
						hs.append(h[:, layer-1])
				else:
					h = sess.run(val_quad_states,
								 feed_dict={val_quad_context: ablated_chunk[j:j+args.n_gpu]})
					hs.append(h[:, layer-1])
		else:
			h = sess.run(val_states,
						 feed_dict={val_context: ablated_chunk[i:i+nbatch]})
			hs.append(h[:, layer-1])
	hs = np.concatenate(hs)
	# hs = np.swapaxes(hs, 0, 1)
	# hs = np.swapaxes(np.array(hs), 0, 1)
	return hs[:-args.cl, args.cl]

def get_hs(chunk):
	nwords = chunk.shape[0]
	nbatch = args.val_batch_size * args.n_gpu
	hs = []
	for i in range(0, nwords, nbatch):
		if i+nbatch >= nwords:
			for j in range(i, nwords, args.n_gpu):
				if j+args.n_gpu >= nwords:
					for k in range(j, nwords):
						h = sess.run(val_single_states,
									 feed_dict={val_single_context: [chunk[k]]})
						hs.append(h)
				else:
					h = sess.run(val_quad_states,
								 feed_dict={val_quad_context: chunk[j:j+args.n_gpu]})
					hs.append(h)
		else:
			h = sess.run(val_states,
						 feed_dict={val_context: chunk[i:i+nbatch]})
			hs.append(h)
	hs = np.concatenate(hs)
	hs = np.swapaxes(hs, 0, 1)
	return np.hstack([hs[:, 0, :args.cl], hs[:, :-args.cl, args.cl]])


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Fine-tune GPT-2 on your custom dataset.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument("--model_name", metavar="MODEL", type=str, default="117M_ft", help="Pretrained model name")
	parser.add_argument("--models_dir", metavar="PATH", type=str, default="transformers/gpt-2_ft/models", help="Path to models directory")

	parser.add_argument("--batch_size", metavar="SIZE", type=int, default=8, help="Batch size")
	parser.add_argument("--accumulate_gradients", metavar="N", type=int, default=1, help="Accumulate gradients across N minibatches.")
	parser.add_argument("--only_train_transformer_layers", default=False, action="store_true", help="Restrict training to the transformer blocks.")

	parser.add_argument("--restore_from", type=str, default="trial6_gpu", help="Either 'latest', 'fresh', or a path to a checkpoint file")
	parser.add_argument("--run_name", type=str, default="", help="Run id. Name of subdirectory in checkpoint/")
	parser.add_argument("--save_every", metavar="N", type=int, default=100, help="Write a checkpoint every N steps")

	parser.add_argument("--val_batch_size", metavar="SIZE", type=int, default=8, help="Batch size for validation.")
	parser.add_argument("--val_batch_count", metavar="N", type=int, default=8, help="Number of batches for validation.")
	parser.add_argument("--val_every", metavar="STEPS", type=int, default=5, help="Calculate validation loss every STEPS steps.")

	parser.add_argument("--pretrain", default=False, action="store_true", help="Load original GPT2 pretrained model.")

	parser.add_argument('--n_gpu', type=int, default=4)
	parser.add_argument('--n_iter', type=int, default=200)

	parser.add_argument("--train", default=False, action="store_true", help="Report perplexity of trained model.")
	parser.add_argument("--test", default=False, action="store_true", help="Report perplexity of trained model.")
	parser.add_argument("--extract_feat", default=False, action="store_true", help="Extract hidden states.")
	parser.add_argument("--extract_ablated_feat", default=False, action="store_true", help="Extract hidden states.")	
	parser.add_argument("--fmri_set", default=False, action="store_true", help="Extract 5 (instead of 15) sessions of fMRI stimulus")
	parser.add_argument('--layer', type=int)
	parser.add_argument('--cl', type=int, default=10)
	parser.add_argument("--save_story_ppx", default=False, action="store_true", help="Report perplexity of trained model.")

	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--lr_warmup', type=float, default=0.002)
	parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
	parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer. <adam|sgd>.")
	parser.add_argument('--b1', type=float, default=0.9)
	parser.add_argument('--b2', type=float, default=0.999)
	parser.add_argument('--e', type=float, default=1e-8)
	parser.add_argument('--max_grad_norm', type=int, default=1)
	parser.add_argument('--l2', type=float, default=0.01)
	parser.add_argument('--vector_l2', action='store_true')
	parser.add_argument('--max_patience', type=int, default=3)

	args = parser.parse_args()

	if args.test or args.extract_feat or args.extract_ablated_feat:
		args.run_name = args.restore_from

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	CHECKPOINT_DIR = os.path.join(user_home_dir, args.models_dir, args.model_name, "checkpoint")

	hparams = gpt2_model.default_hparams()
	with open(os.path.join(user_home_dir, args.models_dir, args.model_name, "hparams.json")) as f:
		hparams.override_from_dict(json.load(f))

	train_chunks, val_chunks, test_chunks = load_dataset(args.models_dir, args.model_name, hparams.n_vocab)

	#########################################################
	## Create training sampler and initialize training ops ##
	#########################################################
	data_sampler = Sampler(train_chunks)
	n_batch_train = args.batch_size * args.n_gpu
	n_train = sum([chunk.shape[0]//hparams.n_ctx for chunk in train_chunks])
	n_updates_total = (n_train//n_batch_train)*args.n_iter

	train_context = tf.placeholder(tf.int32, [n_batch_train, hparams.n_ctx])
	train, train_losses = mgpu_train(train_context, hparams)
	train_loss = tf.reduce_mean(train_losses)
	summary_loss = tf.summary.scalar("loss", train_loss)
	summary_lr = tf.summary.scalar("learning_rate", args.lr)
	summaries = tf.summary.merge([summary_lr, summary_loss])

	#############################################################
	## Create validation sampler and initialize validation ops ##
	#############################################################
	val_data_sampler = Sampler(val_chunks, seed=1)
	val_batches = [[val_data_sampler.sample(hparams.n_ctx) for _ in range(args.val_batch_size*args.n_gpu)]
				   for _ in range(args.val_batch_count)]

	val_context = tf.placeholder(tf.int32, [args.val_batch_size*args.n_gpu, None])
	val_losses, val_ppx, val_states = mgpu_predict(val_context, hparams)
	val_loss = tf.reduce_mean(val_losses)
	val_loss_summary = tf.summary.scalar("val_loss", val_loss)

	######################################
	## Create session, logger and saver ##
	######################################
	params = find_trainable_variables('model')
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	sess.run(tf.global_variables_initializer())

	summary_log = tf.summary.FileWriter(
		os.path.join(CHECKPOINT_DIR, args.run_name))
	saver = tf.train.Saver(
		var_list=params,
		max_to_keep=5)

	############################################################
	## Load either pretrained GPT-2 or fine-tuned checkpoint. ##
	############################################################
	if args.pretrain:
		ckpt = tf.train.latest_checkpoint(
			os.path.join(user_home_dir, args.models_dir, "117M"))
		print("Loading checkpoint", ckpt)
		reader = load_checkpoint(ckpt)

		word_embedding_path = os.path.join(user_home_dir, args.models_dir, args.model_name, "word_embedding.npz")
		word_embedding = np.load(word_embedding_path)['arr_0']
		assert np.all(word_embedding.shape==(hparams.n_vocab, hparams.n_embd))
	else:
		ckpt = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.restore_from))
		print("Loading checkpoint", ckpt)
		reader = load_checkpoint(ckpt)

	for var in params:
		name = var.name.replace(":0", "")
		print(f"Updating {name}")
		if args.pretrain and name == "model/wte":
			sess.run(var.assign(word_embedding))
		elif args.pretrain and name == "model/wpe":
			sess.run(var.assign(reader.get_tensor(name)[:hparams.n_ctx]))
		else:
			sess.run(var.assign(reader.get_tensor(name)))

	if args.train:
		#################################
		## Run training and validation ##
		#################################
		if len(args.run_name) == 0:
			trial = 1
			for f in os.listdir(CHECKPOINT_DIR):
				if "trial" in f and "gpu" in f:
					trial += 1
			args.run_name = "trial%d_gpu"%trial
		maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
		with open(os.path.join(CHECKPOINT_DIR, args.run_name, "config.json"), "w") as f:
			json.dump(vars(args), f)
		counter = 1
		counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, "counter")
		if os.path.exists(counter_path):
			# Load the step number if we"re resuming a run
			# Add 1 so we don"t immediately try to save again
			with open(counter_path, "r") as fp:
				counter = int(fp.read()) + 1

		avg_loss = (0.0, 0.0)
		start_time = time.time()
		best_val_loss = 1000000
		patience = 0
		try:
			while True:
				if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
					print("Calculating validation loss...")
					losses, ppx = [], []
					for batch in tqdm(val_batches):
						l, p = sess.run([val_loss, val_ppx], feed_dict={val_context: batch})
						losses.append(l)
						ppx.append(p)
					v_val_loss = np.mean(losses)
					v_val_ppx = np.array(ppx)
					v_val_ppx = v_val_ppx.sum(0).sum(0)/(args.val_batch_count*args.val_batch_size*args.n_gpu)
					v_val_ppx = np.exp(np.mean(v_val_ppx))
					v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
					summary_log.add_summary(v_summary, counter)
					summary_log.flush()
					print(
						"[{counter} | {time:2.2f}] validation loss = {loss:2.2f} ppx = {ppx:2.2f}"
						.format(
							counter=counter,
							time=time.time() - start_time,
							loss=v_val_loss,
							ppx=v_val_ppx))
					if v_val_loss < best_val_loss:
						save()
						best_val_loss = v_val_loss
						patience = 0
					elif v_val_loss > best_val_loss:
						patience += 1

				(_, t_loss, t_summary) = sess.run(
					(train, train_loss, summaries),
					feed_dict={train_context: sample_batch()})
				summary_log.add_summary(t_summary, counter)
				avg_loss = (avg_loss[0] * 0.99 + t_loss,
							avg_loss[1] * 0.99 + 1.0)
				print(
					"[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}"
					.format(
						counter=counter,
						time=time.time() - start_time,
						loss=t_loss,
						avg=avg_loss[0] / avg_loss[1]))

				counter += 1

				if patience == args.max_patience or counter == args.n_iter:
					save()
					break
		except KeyboardInterrupt:
			print("interrupted")
			save()

	if args.test:
		#####################################################
		## Run model inference and hidden state extraction ##
		#####################################################
		results = {}
		for chunks, mode in zip([val_chunks, test_chunks],
								["val", "test"]):
			split_chunks = np.expand_dims(sample_eval_chunks(chunks, hparams.n_ctx), 1)
			losses, ppx = [], []
			for batch in tqdm(split_chunks):
				l, p = sess.run([val_loss, val_ppx], feed_dict={val_context: batch})
				losses.append(l)
				ppx.append(p)
			v_loss = np.mean(losses)
			v_ppx = np.squeeze(ppx)
			v_ppx = np.exp(np.mean(v_ppx))
			print(
				"{mode} loss = {loss:2.2f} ppx = {ppx:2.2f}"
				.format(
					mode=mode,
					loss=v_loss,
					ppx=v_ppx))
			results["%s_loss"%mode] = str(v_loss)
			results["%s_ppx"%mode] = str(v_ppx)
		with open(os.path.join(CHECKPOINT_DIR, args.restore_from, "results.json"), 'w') as f:
			json.dump(results, f)

	if args.extract_feat:
		val_single_context = tf.placeholder(tf.int32, [1, None])
		val_single_states = mgpu_predict_single(val_single_context, hparams)
		val_quad_context = tf.placeholder(tf.int32, [args.n_gpu, None])
		_, _, val_quad_states = mgpu_predict(val_quad_context, hparams)

		eval_chunks, unk_id = load_eval_dataset(
			args.models_dir, args.model_name, hparams.n_vocab, args.fmri_set)
		cts_chunks = sample_cts_eval_chunks(eval_chunks, hparams.n_ctx, unk_id)
		
		print("Uploading hidden states.")
		save_path = "gpt2_ft/{run}/hidden_state/layer{layer}/cl{cl}/{story}"
		new_save_path = "/mnt/data/angelaz2/gpt2_ft/{run}/hidden_state/layer{layer}/cl{cl}/{story}"
		
		for s, story in enumerate(cts_chunks):
			print("YO IT'S A STORY ")
			print(args.restore_from)
			print(args.layer)
			print(args.cl)
			print(story)
			# if cci.exists_object(
			# 		save_path.format(run=args.restore_from, layer=args.layer, cl=args.cl, story=story)):
			# 	continue
			print('##############################################################')
			print('########################## Story %d ##########################'%(s+1))
			print('##############################################################')
			context_state = get_hs(cts_chunks[story].copy())
			#for layer in range(hparams.n_layer):
			
			if ((story == "persuasionmovement" or story == "persuasionsocial") and args.layer == 9 and args.cl == 10 and args.restore_from == "trial6_gpu"):
				np.save(
					new_save_path.format(run=args.restore_from, layer=args.layer, cl=args.cl, story=story),
					context_state[args.layer])
				#cci.upload_raw_array(
				#	save_path.format(run=args.restore_from, layer=layer+1, cl=args.cl, story=story),
				#	context_state[layer])

	if args.extract_ablated_feat:
		val_single_context = tf.placeholder(tf.int32, [1, None])
		val_single_states = mgpu_predict_single(val_single_context, hparams)
		val_quad_context = tf.placeholder(tf.int32, [args.n_gpu, None])
		_, _, val_quad_states = mgpu_predict(val_quad_context, hparams)

		eval_chunks, unk_id = load_eval_dataset(
			args.models_dir, args.model_name, hparams.n_vocab, args.fmri_set)
		cts_chunks = sample_cts_eval_chunks(eval_chunks, hparams.n_ctx, unk_id)

		print("Uploading ablated hidden states.")
		save_path = "gpt2_ft/{run}/ablated_hidden_state/layer{layer}/cl{cl}/{story}"
		new_save_path = "/mnt/data/angelaz2/gpt2_ft/{run}/ablated_hidden_state/layer{layer}/cl{cl}/{story}"

		for s, story in enumerate(cts_chunks):
			# if cci.exists_object(
			# 		save_path.format(run=args.restore_from, layer=args.layer, cl=args.cl, story=story)):
			# 	continue
			print('##############################################################')
			print('########################## Story %d ##########################'%(s+1))
			print('##############################################################')
			ablated_hs = []
			for pos in tqdm(range(args.cl+2)):
				ablated_hs.append(get_ablated_hs(cts_chunks[story].copy(), pos, args.layer, args.cl)) # [Layers, Words, n_ctx]
			ablated_hs = np.array(ablated_hs)
			print(ablated_hs.shape)
			original_hs = cci.download_raw_array(save_path.format(run=args.restore_from, layer=args.layer, cl=args.cl, story=story).replace("ablated_", ""))
			assert np.allclose(ablated_hs[-1], original_hs[args.cl:], atol=1e-4)

			ref = cci.download_raw_array('transformer/trial1/ablated_hidden_state/layer1/cl%d/%s'%(args.cl, story))
			assert np.all(ablated_hs.shape==ref.shape), (ablated_hs.shape, ref.shape)
			if ((story == "persuasionmovement" or story == "persuasionsocial") and args.layer == 9 and args.cl == 10 and args.restore_from == "trial6_gpu"):
				np.save(
					new_save_path.format(run=args.restore_from, layer=args.layer, cl=args.cl, story=story),
					ablated_hs)
			#cci.upload_raw_array(
			#	save_path.format(run=args.restore_from, layer=args.layer, cl=args.cl, story=story),
			#	ablated_hs)

	if args.save_story_ppx:
		#####################################################
		## ... ##
		#####################################################
		# NOTE: Only runs for 1 GPU and val batch size of 1.

		eval_chunks, unk_id = load_eval_dataset(
			args.models_dir, args.model_name, hparams.n_vocab, fmri_set=False)
		cts_chunks = sample_cts_eval_chunks(eval_chunks, hparams.n_ctx, unk_id)

		save_path = "gpt2_ft/{run}/story_chunks_ppx/original/{story}"
		new_save_path = "/mnt/data/angelaz2/gpt2_ft/{run}/story_chunks_ppx/original/{story}"
		for story in cts_chunks:
			# if cci.exists_object(save_path.format(run=args.restore_from, story=story)): continue
			split_chunks = np.expand_dims(cts_chunks[story], 1)
			losses, ppx = [], []
			for batch in tqdm(split_chunks):
				l, p = sess.run([val_loss, val_ppx], feed_dict={val_context: batch})
				losses.append(l)
				ppx.append(p)
			v_loss = np.mean(losses)
			v_ppx = np.squeeze(ppx)
			print(new_save_path.format(run=args.restore_from, story=story), v_ppx.shape)
			#cci.upload_raw_array(save_path.format(run=args.restore_from, story=story), v_ppx)
			if ((story == "persuasionmovement" or story == "persuasionsocial") and args.restore_from == "trial6_gpu"):
				np.save(new_save_path.format(run=args.restore_from, story=story), v_ppx)

		save_path = "gpt2_ft/{run}/story_chunks_ppx/ablated_pos{pos}/{story}"
		new_save_path = "/mnt/data/angelaz2/gpt2_ft/{run}/story_chunks_ppx/ablated_pos{pos}/{story}"
		for story in cts_chunks:
			print(f"####### Story {story} #######")
			for pos in range(args.cl+2):
				print(f"		####### Ablation position {pos} #######")
				# if cci.exists_object(save_path.format(run=args.restore_from, pos=pos, story=story)): continue
				ablated_split_chunks  = np.expand_dims(cts_chunks[story].copy(), 1)
				if pos <= args.cl:
					ablated_split_chunks[:, :, pos] = unk_id
				losses, ppx = [], []
				for batch in ablated_split_chunks:
					l, p = sess.run([val_loss, val_ppx], feed_dict={val_context: batch})
					losses.append(l)
					ppx.append(p)
				v_loss = np.mean(losses)
				v_ppx = np.squeeze(ppx)
				print(new_save_path.format(run=args.restore_from, pos=pos, story=story), v_ppx.shape)
				#cci.upload_raw_array(save_path.format(run=args.restore_from, pos=pos, story=story), v_ppx)
				if ((pos == 0 or pos == 1 or pos == 2) and (story == "persuasionmovement" or story == "persuasionsocial") and args.restore_from == "trial6_gpu"):
					np.save(new_save_path.format(run=args.restore_from, pos=pos, story=story), v_ppx)
