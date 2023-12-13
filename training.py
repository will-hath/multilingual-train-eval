import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset



from transformers import AutoTokenizer, AutoModel
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

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

    def __call__(self, data):
        tokens_and_mask = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        model_output = self.model(tokens_and_mask["input_ids"], attention_mask=tokens_and_mask["attention_mask"])
        embedding = embedding_model.average_pool(model_output.last_hidden_state, attention_mask=tokens_and_mask["attention_mask"])
        # normalize the embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding    
    
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, native_static, native_dynamic, foreign_dynamic, dynamic_scaling=10e-2):
        # We want the native_static and native_dynamic to be close together
        native_sim = F.cosine_similarity(native_static, native_dynamic)
        dynamic_sim = F.cosine_similarity(native_dynamic, foreign_dynamic)
        loss_contrastive = torch.mean(1 - native_sim) + dynamic_scaling * torch.mean(1 - dynamic_sim)  

        return loss_contrastive


def evaluate(dynamic_model, static_model, dataset, batch_size=32, langs=None, device=device):
    """
    Evaluates the model on the given dataset
    Returns the average loss
    """
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    inter_model_loss_total = 0
    intra_lang_loss_dynamic_total = 0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        # Forward pass
        static_native = static_model(batch["translation"][langs[0]])
        dynamic_native = dynamic_model(batch["translation"][langs[0]])
        dynamic_foreign = dynamic_model(batch["translation"][langs[1]])
        
        inter_model_loss = torch.mean(F.cosine_similarity(static_native, dynamic_native))
        intra_lang_loss_dynamic = torch.mean(F.cosine_similarity(dynamic_native, dynamic_foreign))
        
        inter_model_loss_total += inter_model_loss.item()
        intra_lang_loss_dynamic_total += intra_lang_loss_dynamic.item()
        num_batches += 1

    avg_inter_model_loss = inter_model_loss_total / num_batches
    avg_intra_lang_loss_dynamic = intra_lang_loss_dynamic_total / num_batches

    return avg_inter_model_loss, avg_intra_lang_loss_dynamic        

def train(static_model, dynamic_model, train_dataset, test_dataset, num_text_pairs, evals=[], lr=.0001, dynamic_scaling=.2, batch_size=32, criterion=ContrastiveLoss(), langs=[], device=device):
    """
    Lang[0] is the native language
    Lang[1] is the foreign language
    """
    
    # Define the optimizer
    optimizer = torch.optim.Adam(dynamic_model.model.parameters(), lr=lr)
    
    # Convert the dataset to torch format and create a DataLoader
    train_dataset = train_dataset.with_format("torch")
    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    for i, batch in tqdm(enumerate(dataloader)):
        # Forward pass
        static_native = static_model(batch["translation"][langs[0]])
        dynamic_native = dynamic_model(batch["translation"][langs[0]])
        dynamic_foreign = dynamic_model(batch["translation"][langs[1]])
        loss = criterion(static_native, dynamic_native, dynamic_foreign, dynamic_scaling=dynamic_scaling)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i * batch_size > num_text_pairs:
            break
        
        if i+1 % 500 == 0:
            print(f'Epoch [{i+1}/{num_text_pairs/batch_size}]')
            print(f'Loss: {loss.item()}')
            test = evaluate(dynamic_model, static_model, test_dataset, batch_size=batch_size, langs=langs, device=device)
            evals.append(test)
        # save it at the halfway point
        if i == num_text_pairs // batch_size // 2:
            dynamic_model.model.save_pretrained(f"models/training/e5-base-v2-{langs[1]}-{i * batch_size}-{lr}-{dynamic_scaling}")
        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}') 
    # save the model
    dynamic_model.model.save_pretrained(f"models/trained/e5-base-v2-{langs[1]}-{num_text_pairs}-{lr}-{dynamic_scaling}")
    # save the evals as a text file
    with open(f"models/trained/e5-base-v2-{langs[1]}-{num_text_pairs}-{lr}-{dynamic_scaling}-evals.txt", "w") as f:
        f.write(str(evals))
    return loss.item(), evals
 

if __name__ == "__main__":
    
    language_pairs = ["deu_Latn-eng_Latn", "eng_Latn-swh_Latn"]
    # load in all models

    # native_lang = "eng_Latn"
    # foreign_lang = "spa_Latn"
    lr = .0001
    dynamic_scaling = .4
    # for lr in lrs:
    #     for dynamic_scaling in dynamic_scalings:
    for language_pair in language_pairs:
        print(language_pair)
        test_dataset = load_dataset("allenai/nllb", f"{language_pair}", split="train", streaming=True).take(1000)
        train_dataset = load_dataset("allenai/nllb", f"{language_pair}", split="train", streaming=True).skip(1000)

        parent_model = "e5-base-v2"
        static = embedding_model(AutoModel.from_pretrained("models/raw_models/e5-base-v2"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=True)
        dynamic = embedding_model(AutoModel.from_pretrained("models/raw_models/e5-base-v2"), AutoTokenizer.from_pretrained("models/tokenizers/e5-base-v2"), device=device, inference=False)

        # put into dictionary
        models = {
            "static": static,
            "dynamic": dynamic
        }
        train(static, dynamic, train_dataset, test_dataset, dynamic_scaling=dynamic_scaling, lr=lr, num_text_pairs=400_000, batch_size=64, langs=language_pair.split("-"), device=device)
            
# e5-base-v2-spa_Latn-100000-0.0001-0.4