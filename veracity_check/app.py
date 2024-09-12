from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import numpy as np
import json
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers


app = Flask(__name__)

class BERT_Arch(nn.Module):
    def __init__(self, bert):  
      super(BERT_Arch, self).__init__()
      self.bert = bert   
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass  
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x
    
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)


@app.route('/about', methods=['GET'])
def about():
  return render_template("about.html")

@app.route('/', methods=['GET'])
def index():
  #news=request.form['input']
  return render_template("index.html")

@app.route('/generated', methods=['POST'])
def generate():
  news=request.form['input']
  path = 'c3_new_model_weights.pt'
  model.load_state_dict(torch.load(path))
  MAX_LENGHT = 15
  tokens_unseen = tokenizer.batch_encode_plus(
        [news],
        max_length = MAX_LENGHT,
        pad_to_max_length=True,
        truncation=True
    )
  unseen_seq = torch.tensor(tokens_unseen['input_ids'])
  unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

  with torch.no_grad():
    preds = model(unseen_seq, unseen_mask)
    preds = preds.detach().cpu().numpy()
  preds = np.argmax(preds, axis = 1)
  print(news)
  if preds[0] == 1:
    out = "The news provided is Fake"
  else:
    out = "The news provided is Real"
    
  return render_template("index.html", output = out, input = news)

@app.route('/team', methods=['GET'])
def team():
  return render_template("team.html")

@app.route('/chatbot', methods=['GET'])
def chatbot():
  
  return render_template("chatbot.html")

@app.route('/chatbot1', methods=['POST'])
def chatbot1():
  query = request.form['userInput']
  document_store = FAISSDocumentStore.load("news_faiss")
  """document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
  doc_dir = "/home/sunil/WebScraper/coppellisd/schools/text_files"

  docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

  document_store.write_documents(docs)"""
  retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
  document_store.update_embeddings(retriever)
  reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
  pipe = ExtractiveQAPipeline(reader, retriever)
  prediction = pipe.run(
    query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
  print_answers(prediction, details="minimum")
  final_output = {}
  for i in range(len(prediction['answers'])):
      res = f"Result {i}"
      final_output[res] = [
          {
              'Answer':prediction['answers'][i].answer,
              'Context':prediction['answers'][i].context
          }
      ]
     
  out = json.dumps(final_output)

  
  return render_template("chatbot.html", output = out, input = query)  

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
