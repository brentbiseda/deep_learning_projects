"""Filename: theroom.py
"""
import sys
import os
import html
import pandas as pd
from flask import Flask, Response, request, render_template, send_from_directory, url_for
from fastai.text import *
from pathlib import Path
import pickle
import numpy as np
import collections

PATH=Path('the_room/')
CLAS_PATH=Path('the_room/the_room_clas/')
LM_PATH=Path('the_room/the_room_lm/')

trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')

itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))
stoi = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos)})

vs=len(itos)
vs,len(trn_lm)

wd=1e-7
bptt=70
bs=10

opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
em_sz,nh,nl = 400,1150,3

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*2.25

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

learner= md.get_model(opt_fn, em_sz, nh, nl, dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.load('lm1')
learner.load_encoder('lm1_enc')

m=learner.model


app = Flask(__name__)

def sample_model(m, stoi, itos, bs, s, speaker_name):
	s = s + " " + speaker_name + " : "

	s_toks = Tokenizer().proc_text(s)
	s_nums = [stoi[i] for i in s_toks]
	s_nums = [i for i in s_nums if i != -1]
	s_var = V(np.array(s_nums))[None]
	
	m[0].bs = 1
	m.eval()
	m.reset()
	res, *_ = m(s_var)
	words = ""
	flag = True
	
	while flag == True:
		r = torch.multinomial(res[-1].exp(), 2)
		if r.data[0] == 0:
			r = r[1]
		else:
			r = r[0]
		word = itos[to_np(r)[0]]
		res, *_ = m(r[0].unsqueeze(0))
		if word == "xeos":
			flag = False
		else:
			words += word + ' '
	m[0].bs = bs
	return (words)


if __name__ == "__main__":
	app.run(host='127.0.0.1')

@app.route('/')
def my_form():
	return render_template('my-form.html')


@app.route('/', methods=['POST'])
def my_form_post():
	text = request.form['text']
	question = text.lower()
	speaker = "johnny"
	response = "poop"
	response = speaker + ": " + sample_model(m, stoi, itos, bs, question, speaker)
	
	return render_template('my-form.html', response = response, question = question)