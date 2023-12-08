import os, sys
import numpy as np
from pathlib import Path
import tqdm
import json
import tensorflow.compat.v1 as tf
import cottoncandy as cc

user_home_dir = str(Path.home())
sys.path.insert(0, os.path.join(user_home_dir, 'transformers/phrase-level-models'))
import phrase_level_model_utils as plm_utils
cci_data = cc.get_interface('neural-model-data', verbose=False)

def get_story_txt_splits():
	data_path = os.path.join(user_home_dir, 'transformers/gpt-2_ft/new_data')
	with open(os.path.join(data_path, 'story_txts.json'), 'r') as f:
		story_txts = json.load(f)
	stories = np.sort(list(story_txts.keys()))

	_, val_stories, test_stories = plm_utils.get_stories_in_sessions(list(map(str, range(1, 16))))
	extra_test_stories, _, _ = plm_utils.get_stories_in_sessions(list(map(str, range(16, 21))))
	test_stories = test_stories + extra_test_stories
	val_stories = [story for story in val_stories if "canplanetearthfeedtenbillionpeople" not in story]
	val_stories = val_stories + ["canplanetearthfeedtenbillionpeople"]
	assert len(set(val_stories) & set(test_stories)) == 0
	train_stories = list((set(stories) - set(val_stories)) - set(test_stories))
	assert len(set(train_stories) & set(val_stories)) == 0
	assert len(set(train_stories) & set(test_stories)) == 0
	assert len(train_stories) + len(val_stories) + len(test_stories) == len(stories), (len(train_stories) + len(val_stories) + len(test_stories), len(stories))
	print("Train stories: %d | Val stories: %d | Test stories: %d"%(len(train_stories), len(val_stories), len(test_stories)))

	train_story_txts = {story: story_txts[story] for story in train_stories}
	val_story_txts = {story: story_txts[story] for story in val_stories}
	test_story_txts = {story: story_txts[story] for story in test_stories}

	return train_story_txts, val_story_txts, test_story_txts

def get_token_chunks(story_txts, word2int):
	token_chunks = []
	for story in story_txts:
		tokens = np.array([word2int.get(word, word2int["<unk>"]) \
						   for word in story_txts[story]])
		token_chunks.append(tokens)
	return token_chunks

def load_dataset(models_dir, model_name, vocab_size):
	vocab_path = os.path.join(user_home_dir, models_dir, model_name, "vocab.npz")
	vocab = np.load(vocab_path)['arr_0']
	assert np.all(vocab==np.sort(vocab))
	assert len(vocab) == vocab_size
	word2int = {word: i for i, word in enumerate(vocab)}

	train_story_txts, val_story_txts, test_story_txts = get_story_txt_splits()
	train_chunks = get_token_chunks(train_story_txts, word2int)
	val_chunks = get_token_chunks(val_story_txts, word2int)
	test_chunks = get_token_chunks(test_story_txts, word2int)
	return train_chunks, val_chunks, test_chunks

def load_eval_dataset2(models_dir, model_name, vocab_size, fmri_set):
	vocab_path = os.path.join(user_home_dir, models_dir, model_name, "vocab.npz")
	vocab = np.load(vocab_path)['arr_0']
	assert np.all(vocab==np.sort(vocab))
	# assert len(vocab) == vocab_size
	word2int = {word: i for i, word in enumerate(vocab)}

	gpt_vocab_file = 'vocab_70stim+top10k'
	gpt_vocab = np.array(cci_data.download_raw_array(gpt_vocab_file), str)
	gpt_int2word = {i: gpt_vocab[i] for i in range(len(gpt_vocab))}
	
	if fmri_set:
		last_session = 5
	else:
		last_session = 15
	print("Last session: %d"%last_session)
	eval_stories, _, _ = plm_utils.get_stories_in_sessions(list(map(str, range(1, last_session+1))))
	print("Eval stories: %d"%(len(eval_stories)))

	eval_chunks = {}
	for story in eval_stories:
		gpt_tokens = cci_data.download_raw_array('stimulus_seqs101/%s'%story)[:, 0]
		words = list(map(gpt_int2word.get, gpt_tokens))
		gpt2_tokens = np.array([word2int.get(word, word2int["<unk>"]) for word in words])
		assert gpt2_tokens.shape[0]==gpt2_tokens.shape[0]
		eval_chunks[story] = gpt2_tokens
	return eval_chunks, word2int["<unk>"]

def load_eval_dataset(models_dir, model_name, vocab_size, fmri_set):
	vocab_path = os.path.join(user_home_dir, models_dir, model_name, "vocab.npz")
	vocab = np.load(vocab_path)['arr_0']
	assert np.all(vocab==np.sort(vocab))
	# assert len(vocab) == vocab_size
	word2int = {word: i for i, word in enumerate(vocab)}

	gpt_vocab_file = 'vocab_70stim+top10k'
	gpt_vocab = np.array(cci_data.download_raw_array(gpt_vocab_file), str)
	gpt_int2word = {i: gpt_vocab[i] for i in range(len(gpt_vocab))}

	eval_chunks = {}
	story_words = {
		"persuasionmovement": "The red triangle positions itself in the entryway of the box The red triangle comes back into the box \
		on top of the blue triangle The red triangle rotates touching the blue triangle The red triangle and blue triangle move towards \
		the entryway of the box Once the blue triangle reaches the entryway the red triangle pivots so its point is attached to the side \
		of the blue triangle that is fully in the box The red triangle moves back into the box then touches the blue triangle again The \
		blue triangle rotates clockwise then moves until one of its sides is along one of the sides of the red triangle The red triangle \
		and blue triangle move diagonally until the blue triangle is near the outside of the box The blue triangle separates and moves \
		towards the right whilst the red triangle rotates so that one of its sides nearly aligns with the entryway The blue triangle \
		rotates until its tip is facing up and slightly to the left then moves that direction It then continues to rotate clockwise \
		and move slightly in each position The red triangle then begins to rotate clockwise As the blue triangle moves towards the \
		bottom of the screen the red triangle slowly moves towards it until its tip is connected with the side of the blue triangle \
		The blue triangle turns in a circle clockwise until its tip touches the side of the red triangle The red triangle moves back \
		so its tip touches the tip of the blue triangle The triangles briefly separate before coming together at their tips so they are \
		aligned along an axis The two triangles then turn in a circle",
		"persuasionsocial": "The red triangle decides the blue triangle needs a little push to move out of the box The red triangle \
		pushes the blue triangle outside the box and the blue triangle complies Once the triangles are in the entryway the red triangle \
		aligns itself to best give a final push with its tip nudging the blue triangle The blue triangle is still not ready and turns \
		back towards the red triangle wanting more comfort The red triangle pushes the blue triangle back out of the box again The red \
		triangle blocks the entrance to force the blue triangle to stay outside The blue triangle shakes a little bit in fear before \
		looking towards the sky trying to calm itself and adjust to its new environment It then starts exploring and moving around a little \
		because it is curious The red triangle deems that the blue triangle is comfortable and unblocks the opening The red triangle moves \
		to catch up and talk with the blue triangle The blue triangle is happy and the two triangles align tips and spin around to celebrate",
	}
	for story in story_words:
		words = story_words[story].split(" ")
		gpt2_tokens = np.array([word2int.get(word, word2int["<unk>"]) for word in words])
		eval_chunks[story] = gpt2_tokens
	return eval_chunks, word2int["<unk>"]

def load_avaidya_dataset(models_dir, model_name, vocab_size):
	vocab_path = os.path.join(user_home_dir, models_dir, model_name, "vocab.npz")
	vocab = np.load(vocab_path)['arr_0']
	assert np.all(vocab==np.sort(vocab))
	# assert len(vocab) == vocab_size
	word2int = {word: i for i, word in enumerate(vocab)}

	gpt_vocab_file = 'vocab_70stim+top10k'
	gpt_vocab = np.array(cci_data.download_raw_array(gpt_vocab_file), str)
	gpt_int2word = {i: gpt_vocab[i] for i in range(len(gpt_vocab))}

	eval_chunks = {}
	story_words = {
		"wheretheressmoke": "we start to trade stories about our lives we're both from up north we're both kind of newish to the \
							neighborhood this is in florida we both went to college not great colleges but man we graduated and \
							i'm actually finding myself a little jealous of her because she has this really cool job washing dogs \
							she had horses back home and she really loves we start to trade stories about our lives we're both \
							from up north we're both kind of newish to the neighborhood this is in florida we both went to college \
							not great colleges but man we graduated and i'm actually finding myself a little jealous of her because \
							she has this really cool job washing dogs she had horses back home and she really loves we start to trade \
							stories about our lives we're both from up north we're both kind of newish to the neighborhood this is \
							in florida we both went to college not great colleges but man we graduated and i'm actually finding myself \
							a little jealous of her because she has this really cool job washing dogs she had horses back home and she really loves we",
		"fromboyhoodtofatherhood": "get out to the hamptons and we're at this farmhouse and it was like a scene out of christopher isherwood the berlin stories all these blonde boys about ten of us running around doing push ups so that our muscles would swell and in and out of the pool and a big buffet and everything waiting for the light to change get out to the hamptons and we're at this farmhouse and it was like a scene out of christopher isherwood the berlin stories all these blonde boys about ten of us running around doing push ups so that our muscles would swell and in and out of the pool and a big buffet and everything waiting for the light to change get out to the hamptons and we're at this farmhouse and it was like a scene out of christopher isherwood the berlin stories all these blonde boys about ten of us running around doing push ups so that our muscles would swell and in and out of the pool and a big buffet and everything waiting for the light to change get",
		"onapproachtopluto": "nine hours i find myself nine hours later back in the situation room looking through the glass window at the operations people hoping this works when i see people start cheering and erupting in cheers and excited and i hear alice bowman's voice over the intercom we are back on the prime nine hours i find myself nine hours later back in the situation room looking through the glass window at the operations people hoping this works when i see people start cheering and erupting in cheers and excited and i hear alice bowman's voice over the intercom we are back on the prime nine hours i find myself nine hours later back in the situation room looking through the glass window at the operations people hoping this works when i see people start cheering and erupting in cheers and excited and i hear alice bowman's voice over the intercom we are back on the prime nine hours i find myself nine",
	}
	for story in story_words:
		words = story_words[story].split(" ")
		gpt2_tokens = np.array([word2int.get(word, word2int["<unk>"]) for word in words])
		eval_chunks[story] = gpt2_tokens
	return eval_chunks, word2int["<unk>"]