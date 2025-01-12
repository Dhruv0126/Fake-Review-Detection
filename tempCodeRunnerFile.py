import pickle 
from flask import Flask, request, jsonify
import string
from nltk.corpus import stopwords
from flask_cors import CORS
