import torch
from mteb import MTEB
from tasks import *
from transformers import AutoTokenizer, AutoModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# load from disk
class embedding_model:
    def __init__(self, model, tokenizer, device, inference=False):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        if inference:
            self.model.eval()
            self.model.requires_grad_(False)

    def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def encode(self, sentences, batch_size=512):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            embedding = self.__call__(batch_sentences)
            # Add each sentence embedding to the list as a separate ndarray
            embeddings.extend(embedding.detach().cpu().numpy())
        return embeddings

    def __call__(self, data):
        tokens_and_mask = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        model_output = self.model(tokens_and_mask["input_ids"], attention_mask=tokens_and_mask["attention_mask"])
        embedding = embedding_model.average_pool(model_output.last_hidden_state, attention_mask=tokens_and_mask["attention_mask"])
        # normalize the embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding    


if __name__ == "__main__":
    # lrs = [.001, .0001, .00001]
    # dynamic_scalings = [4e-1, 1e-1, 1e-2]
    # for lr in lrs:
    #     for dynamic_scaling in dynamic_scalings:
    # tests = [("e5-base-v2-deu_Latn-400000-0.0001-0.4", GermanRedditClustering()),
    #           ("e5-base-v2-spa_Latn-400000-0.0001-0.4", SpanishRedditClustering()), 
    #           ("e5-base-v2-tur_Latn-400000-0.0001-0.4", TurkishRedditClustering()), 
    #           ("e5-base-v2-swh_Latn-400000-0.0001-0.4", SwahiliRedditClustering()),
    #           ("e5-base-v2-fra_Latn-400000-0.0001-0.4", FrenchRedditClustering()),
    #           ("e5-base-v2-deu_Latn-400000-0.0001-0.4", "RedditClustering"),
    #           ("e5-base-v2-spa_Latn-400000-0.0001-0.4", "RedditClustering"), 
    #           ("e5-base-v2-tur_Latn-400000-0.0001-0.4", "RedditClustering"), 
    #           ("e5-base-v2-swh_Latn-400000-0.0001-0.4", "RedditClustering"),
    #           ("e5-base-v2-fra_Latn-400000-0.0001-0.4", "RedditClustering")]
    # test raw_models/multilingual-e5-base on all languages
    tests = [("multilingual-e5-base", GermanRedditClustering()),
                ("multilingual-e5-base", SpanishRedditClustering()),
                ("multilingual-e5-base", TurkishRedditClustering()),
                ("multilingual-e5-base", SwahiliRedditClustering()),
                ("multilingual-e5-base", FrenchRedditClustering())]
    
    for model_name, task in tests:
        print(f"Running {model_name}")
        print(f"Running {task}")
        model = embedding_model(AutoModel.from_pretrained(f"models/trained/{model_name}"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=True)
        evaluation = MTEB(tasks=[task], device=device, verbose=3)
        evaluation.run(model, output_folder=f"results/{model_name}/", overwrite_results=False, verbose=3)
    # model = embedding_model(AutoModel.from_pretrained(f"models/trained/{model_name}"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=True)
    # evaluation = MTEB(tasks=["reddit-clustering"], device=device, verbose=3)
    # evaluation.run(model, output_folder=f"results/{model_name}/", overwrite_results=False, verbose=3)

# load in all models
# dyn = embedding_model(AutoModel.from_pretrained("models/trained/e5-base-v2-esp-150k"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=True)
# other = embedding_model(AutoModel.from_pretrained("models/training/e5-base-v2-spa_Latn-96000"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=True)
# sta = embedding_model(AutoModel.from_pretrained("models/raw_models/e5-base-v2"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=True)
# mult = embedding_model(AutoModel.from_pretrained("models/raw_models/multilingual-e5-small"), AutoTokenizer.from_pretrained("models/tokenizers/multilingual-e5-small"), device=device, inference=True)
# multilingual_model = embedding_model(AutoModel.from_pretrained("intfloat/multilingual-e5-base"), AutoTokenizer.from_pretrained("models/tokenizers/multilingual-e5-small"), device=device, inference=True)

# # put into dictionary
# models = {
#     "dyn": dyn,
#     "sta": sta,
#     "mult": mult,
#     ".01": other
# }

# model_name = ".01"
# model = models[model_name]
# model_name = "e5-base-v2-spa_Latn-100000-0.0001-0.4"
# evaluation = MTEB(tasks=[SpanishRedditClustering()], device=device, verbose=3)
# evaluation.run(multilingual_model, output_folder=f"results/{model_name}/", overwrite_results=False, verbose=3)







