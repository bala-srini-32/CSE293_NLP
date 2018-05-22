import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import io
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Word Translation Without Parallel Data')
parser.add_argument("-s","--src", type=str, required=True, help="Source embeddings vec file")
parser.add_argument("-t","--target", type=str, required=True, help="Target embeddings vec file")
inp = params = parser.parse_args()
assert os.path.isfile(inp.src)
assert os.path.isfile(inp.target)
source_text_file = inp.src
target_text_file = inp.target


vocab_length = 200000
dimension = 300
def load_pretrained_embeddings(filepath):
# load pretrained embeddings
    count = 0
    word_to_id_lang = {}
    id_to_word_lang = {}
    vectors_lang = []
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                header = line.split()
                assert len(header) == 2
                assert dimension == int(header[1])
            else:
                if i%2000 == 0 :
                        print("Reading %s file %0.2f percent done"%(filepath,float(i)/2000.0))
                word, vector_rep = line.rstrip().split(' ', 1)
                word = word.lower()
                vector_rep = np.fromstring(vector_rep, sep=' ')
                if vector_rep.shape[0] == dimension :
                    if word not in word_to_id_lang :
                        word_to_id_lang[word] = count
                        vectors_lang.append(vector_rep[None])
                        count+=1
                    if count == vocab_length :
                        break

                else:
                    continue
    id_to_word_lang = {value: key for key, value in word_to_id_lang.items()}
    embeddings = np.concatenate(vectors_lang, 0)
    return word_to_id_lang,id_to_word_lang,embeddings



src_word_to_id,src_id_to_word,src_embeddings = load_pretrained_embeddings(source_text_file)
target_word_to_id,target_id_to_word,target_embeddings = load_pretrained_embeddings(target_text_file)
src_embedding_learnable = nn.Embedding(len(src_word_to_id),dimension)
src_embedding_learnable.weight = nn.Parameter(torch.from_numpy(src_embeddings).float())
target_embedding_learnable = nn.Embedding(len(target_word_to_id),dimension)
target_embedding_learnable.weight = nn.Parameter(torch.from_numpy(target_embeddings).float())



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.linear1 = nn.Linear(300,2048)
        self.linear2 = nn.Linear(2048,2048)
        self.linear3 = nn.Linear(2048,1)
        
    def forward(self,x):
        x = self.linear1(F.dropout(x,p=0.1))
        x = self.linear2(F.leaky_relu(x,negative_slope=0.2))
        x = self.linear3(F.leaky_relu(x,negative_slope=0.2))
        x = F.sigmoid(x)
        return x.view(-1)
    
class Mapper(nn.Module):
    def __init__(self):
        super(Mapper,self).__init__()
        self.linear1 = nn.Linear(300,300)
    
    def forward(self,x):
        x = self.linear1(x)
        return x




discriminator = Discriminator()
mapper = Mapper()
criterion1 = nn.BCELoss()
optimizer1 = optim.SGD(discriminator.parameters(), lr=0.1)
criterion2 = nn.BCELoss()
optimizer2 = optim.SGD(mapper.parameters(), lr=0.1)




def avg_10_distance(emb2,emb1):
    size = 5000
    all_distances = []
    emb = emb.t().contiguous()
    for i in range(0, emb1.shape[0], size):
        distances = emb1[i:i + size].mm(emb2)
        best_distances, _ = distances.topk(10, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()




def top_words(emb1, emb2):
    #top translation pairs
    size = 5000

    ranked_scores = []
    ranked_targets = []
    num = 50000

    # average distances to 10 nearest neighbors
    average_dist_src = torch.from_numpy(avg_10_distance(emb2, emb1))
    average_dist_target = torch.from_numpy(avg_10_distance(emb2, emb1))
    average_dist_src = average_dist_src.type_as(emb1)
    average_dist_target = average_dist_target.type_as(emb2)

    for i in range(0, num, size):
        scores = emb2.mm(emb1[i:min(num, i + size)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist_src[i:min(num, i + size)][:, None] + average_dist_target[None, :])
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

        # update scores / potential targets
        ranked_scores.append(best_scores.cpu())
        ranked_targets.append(best_targets.cpu())

    ranked_scores = torch.cat(ranked_scores, 0)
    ranked_targets = torch.cat(ranked_targets, 0)

    ranked_pairs = torch.cat([torch.arange(0, ranked_targets.size(0)).long().unsqueeze(1),ranked_targets[:, 0].unsqueeze(1)], 1)

    #Reordering them
    diff = ranked_scores[:, 0] - ranked_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    ranked_scores = ranked_scores[reordered]
    ranked_pairs = ranked_pairs[reordered]

    selected = ranked_pairs.max(1)[0] <= num
    mask = selected.unsqueeze(1).expand_as(ranked_scores).clone()
    ranked_scores = ranked_scores.masked_select(mask).view(-1, 2)
    ranked_pairs = ranked_pairs.masked_select(mask).view(-1, 2)

    return ranked_pairs




def proxy_construct_dictionary(src_emb_map_validation,target_emb_map_validation,src_to_target_dictionary,target_to_src_dictionary):    
    target_to_src_dictionary = torch.cat([target_to_src_dictionary[:, 1:], target_to_src_dictionary[:, :1]], 1)
    src_to_target_dictionary = set([(a, b) for a, b in src_to_target_dictionary])
    target_to_src_dictionary = set([(a, b) for a, b in target_to_src_dictionary])
    final_pairs = src_to_target_dictionary.intersection(target_to_src_dictionary)
    if len(final_pairs) == 0:
        return None
    dictionary = torch.Tensor(list([[a, b] for (a, b) in final_pairs])).long()
    return dictionary




validation_tracker = 0
for epoch in range(10): #5 Epochs 
    for iteration in range(30000):
        # discriminator trained 3 times for every mapping training
        for i in range(3):
            discriminator.train()
            #Set gradient to zero before computation at every step
            optimizer1.zero_grad()
            src_lang_word_id = torch.Tensor(32).random_(50000).long()
            src_lang_word_emb = src_embedding_learnable(src_lang_word_id)
            target_lang_word_id = torch.Tensor(32).random_(50000).long()
            target_lang_word_emb = target_embedding_learnable(target_lang_word_id)
            src_mult_mapper = mapper(src_lang_word_emb)
            input_tensor = torch.cat([src_mult_mapper,target_lang_word_emb],0)
            output_tensor = torch.Tensor(64).zero_().float()
            output_tensor[:32] = 1 -0.2 #Smoothing
            output_tensor[32:] = 0.2
            prediction = discriminator(input_tensor)
            #Compute loss and propogate backward
            loss = criterion1(prediction,output_tensor)
            loss.backward()
            optimizer1.step()

        # mapping training 
        discriminator.eval()
        #Set gradient to zero before computation at every step
        optimizer2.zero_grad()
        src_lang_word_id = torch.Tensor(32).random_(50000).long()
        src_lang_word_emb = src_embedding_learnable(src_lang_word_id)
        target_lang_word_id = torch.Tensor(32).random_(50000).long()
        target_lang_word_emb = target_embedding_learnable(target_lang_word_id)
        src_mult_mapper = mapper(src_lang_word_emb)
        input_tensor = torch.cat([src_mult_mapper,target_lang_word_emb],0)
        output_tensor = torch.Tensor(64).zero_().float()
        output_tensor[:32] = 1 -0.2 #Smoothing
        output_tensor[32:] = 0.2
        prediction = discriminator(input_tensor)
        loss = criterion2(prediction,1-output_tensor)
        loss.backward()
        optimizer2.step()
        mapping_tensor = mapper.linear1.weight.data
        mapping_tensor.copy_((1.01) * mapping_tensor - 0.01 * mapping_tensor.mm(mapping_tensor.t().mm(mapping_tensor)))

        
    #Validation through proxy parralel dictionary construction (both directions) and CSLS
    src_emb_map_validation = mapper(src_embedding_learnable.weight)
    target_emb_map_validation = target_embedding_learnable.weight
    src_to_target_dictionary = top_words(src_emb_map_validation,target_emb_map_validation)
    target_to_src_dictionary = top_words(target_emb_map_validation,src_emb_map_validation)
    dictionary = proxy_construct_dictionary(src_emb_map_validation,target_emb_map_validation,src_to_target_dictionary,target_to_src_dictionary)
    if dictionary is None:
        mean_cosine = -1e9
    else:
        mean_cosine = (src_emb_map_validation[dictionary[:, 0]] * target_emb_map_validation[dictionary[:, 1]]).sum(1).mean()

    # Dampenining by 0.95
    optimizer1.param_groups[0]['lr'] = 0.95*optimizer1.param_groups[0]['lr']
    optimizer2.param_groups[0]['lr'] = 0.95*optimizer2.param_groups[0]['lr']
    #Divide by 2 if validation decreases
    if mean_cosine > 0 and mean_cosine < validation_tracker :
        optimizer1.param_groups[0]['lr'] = 0.5*optimizer1.param_groups[0]['lr']
        optimizer2.param_groups[0]['lr'] = 0.5*optimizer2.param_groups[0]['lr']
        validation_tracker = max(mean_cosine,validation_tracker)
    print(epoch,mean_cosine)




#Just 1 round of refinement - authors say beyond 1 round, gains are minimal
src_refine = src_embedding_learnable.weight.data[dictionary[:, 0]]
target_refine = target_embedding_learnable.weight.data[dictionary[:, 1]]
u,s,v = torch.svd(torch.mm(target_refine,src_refine.t()))
mapper.linear1.weight.copy_(torch.mm(u,v).float())



#Checking the value of validation metric again
#Validation through proxy parralel dictionary construction (both directions) and CSLS
src_emb_map_validation = mapper(src_embedding_learnable.weight)
target_emb_map_validation = target_embedding_learnable.weight
src_to_target_dictionary = top_words(src_emb_map_validation,target_emb_map_validation)
target_to_src_dictionary = top_words(target_emb_map_validation,src_emb_map_validation)
dictionary = proxy_construct_dictionary(src_emb_map_validation,target_emb_map_validation,src_to_target_dictionary,target_to_src_dictionary)
if dictionary is None:
    mean_cosine = -1e9
else:
    mean_cosine = (src_emb_map_validation[dictionary[:, 0]] * target_emb_map_validation[dictionary[:, 1]]).sum(1).mean()
print (mean_cosine)

