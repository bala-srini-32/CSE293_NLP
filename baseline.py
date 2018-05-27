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

src_embedding_learnable = nn.Embedding(len(src_word_to_id),dimension).cuda()
src_embedding_learnable.weight = nn.Parameter(torch.from_numpy(src_embeddings).float())
src_embeddings_norm = src_embedding_learnable.weight/src_embedding_learnable.weight.norm(2, 1, keepdim=True).expand_as(src_embedding_learnable.weight)
src_embedding_learnable.weight.data.copy_(src_embeddings_norm)
target_embedding_learnable = nn.Embedding(len(target_word_to_id),dimension).cuda()
target_embedding_learnable.weight = nn.Parameter(torch.from_numpy(target_embeddings).float())
target_embeddings_norm = target_embedding_learnable.weight/target_embedding_learnable.weight.norm(2, 1, keepdim=True).expand_as(target_embedding_learnable.weight)
target_embedding_learnable.weight.data.copy_(target_embeddings_norm)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.linear1 = nn.Linear(300,2048).cuda()
        self.linear2 = nn.Linear(2048,2048).cuda()
        self.linear3 = nn.Linear(2048,1).cuda()
        
    def forward(self,x):
        x = self.linear1(F.dropout(x,p=0.1)).cuda()
        x = self.linear2(F.leaky_relu(x,negative_slope=0.2)).cuda()
        x = self.linear3(F.leaky_relu(x,negative_slope=0.2)).cuda()
        x = F.sigmoid(x).cuda()
        return x.view(-1)
    
class Mapper(nn.Module):
    def __init__(self):
        super(Mapper,self).__init__()
        self.linear1 = nn.Linear(300,300).cuda()
    
    def forward(self,x):
        x = self.linear1(x).cuda()
        return x




discriminator = Discriminator().cuda()
mapper = Mapper().cuda()
criterion1 = nn.BCELoss().cuda()
optimizer1 = optim.SGD(discriminator.parameters(), lr=0.1)
criterion2 = nn.BCELoss().cuda()
optimizer2 = optim.SGD(mapper.parameters(), lr=0.1)




def avg_10_distance(emb2,emb1):
    size = 128
    all_distances = []
    emb2 = emb2.t().contiguous()
    for i in range(0, emb1.shape[0], size):
        distances = emb1[i:i + size].mm(emb2)
        best_distances, _ = distances.topk(10, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()




def top_words(emb1, emb2):
    #top translation pairs
    size = 64

    ranked_scores = []
    ranked_targets = []
    num = 2000

    # average distances to 10 nearest neighbors
    average_dist_src = torch.from_numpy(avg_10_distance(emb2, emb1))
    average_dist_target = torch.from_numpy(avg_10_distance(emb2, emb1))
    average_dist_src = average_dist_src.type_as(emb1)
    average_dist_target = average_dist_target.type_as(emb2)

    for i in range(0, num, size):
        scores = emb2.mm(emb1[i:min(num, i + size)].t()).t()
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
    dictionary = torch.Tensor(list([[a, b] for (a, b) in final_pairs])).long().cuda()
    return dictionary




validation_tracker = 0
for epoch in range(3): #3 Epochs 
    for iteration in range(30000):
        if iteration % 10 == 0 :
            print("epoch = %d, iteration = %d"%(epoch,iteration))
        # discriminator trained 3 times for every mapping training
        for i in range(3):
            discriminator.train()
            #Set gradient to zero before computation at every step
            optimizer1.zero_grad()
            src_lang_word_id = torch.Tensor(32).random_(50000).long()
            src_lang_word_emb = src_embedding_learnable(src_lang_word_id).cuda()
            target_lang_word_id = torch.Tensor(32).random_(50000).long()
            target_lang_word_emb = target_embedding_learnable(target_lang_word_id).cuda()
            src_mult_mapper = mapper(src_lang_word_emb).cuda()
            input_tensor = torch.cat([src_mult_mapper,target_lang_word_emb],0).cuda()
            output_tensor = torch.Tensor(64).zero_().float().cuda()
            output_tensor[:32] = 1 -0.2 #Smoothing
            output_tensor[32:] = 0.2
            prediction = discriminator(input_tensor).cuda()
            #Compute loss and propogate backward
            loss = criterion1(prediction,output_tensor).cuda()
            loss.backward()
            optimizer1.step()

        # mapping training 
        discriminator.eval()
        #Set gradient to zero before computation at every step
        optimizer2.zero_grad()
        src_lang_word_id = torch.Tensor(32).random_(50000).long()
        src_lang_word_emb = src_embedding_learnable(src_lang_word_id).cuda()
        target_lang_word_id = torch.Tensor(32).random_(50000).long()
        target_lang_word_emb = target_embedding_learnable(target_lang_word_id).cuda()
        src_mult_mapper = mapper(src_lang_word_emb)
        input_tensor = torch.cat([src_mult_mapper,target_lang_word_emb],0).cuda()
        output_tensor = torch.Tensor(64).zero_().float().cuda()
        output_tensor[:32] = 1 -0.2 #Smoothing
        output_tensor[32:] = 0.2
        prediction = discriminator(input_tensor).cuda()
        loss = criterion2(prediction,1-output_tensor).cuda()
        loss.backward()
        optimizer2.step()
        mapping_tensor = mapper.linear1.weight.data
        mapping_tensor.copy_((1.01) * mapping_tensor - 0.01 * mapping_tensor.mm(mapping_tensor.t().mm(mapping_tensor)))

        
    #Validation through proxy parralel dictionary construction (both directions) and CSLS
    src_emb_map_validation = mapper(src_embedding_learnable.weight.cuda()).cuda()
    target_emb_map_validation = target_embedding_learnable.weight.cuda()
    src_emb_map_validation = src_emb_map_validation/src_emb_map_validation.norm(2, 1, keepdim=True).expand_as(src_emb_map_validation)
    target_emb_map_validation = target_emb_map_validation/target_emb_map_validation.norm(2, 1, keepdim=True).expand_as(target_emb_map_validation)
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
src_emb_map_validation = mapper(src_embedding_learnable.weight.cuda())
target_emb_map_validation = target_embedding_learnable.weight
src_to_target_dictionary = top_words(src_emb_map_validation.cuda(),target_emb_map_validation.cuda())
target_to_src_dictionary = top_words(target_emb_map_validation.cuda(),src_emb_map_validation.cuda())
dictionary = proxy_construct_dictionary(src_emb_map_validation,target_emb_map_validation,src_to_target_dictionary,target_to_src_dictionary)
if dictionary is None:
    mean_cosine = -1e9
else:
    mean_cosine = (src_emb_map_validation[dictionary[:, 0]] * target_emb_map_validation[dictionary[:, 1]]).sum(1).mean()
print (mean_cosine)


data_directory = "data/crosslingual/dictionaries/"
source_lang = source_text_file.split(".")[-2]
target_lang = target_text_file.split(".")[-2]

def load_dictionary(path, word2id1, word2id2):
    assert os.path.isfile(path)
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0
    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]
    return dico

print("Task1 : Word Translation")
path = os.path.join(data_directory, '%s-%s.5000-6500.txt' % (source_lang, target_lang))
dico = load_dictionary(path, src_word_to_id, target_word_to_id)
dico = dico.cuda()
emb1 = mapper(src_embedding_learnable.weight.cuda())
emb2 = target_embedding_learnable.weight.cuda()
emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
average_dist1 = avg_10_distance(emb2, emb1)
average_dist2 = avg_10_distance(emb1, emb2)
average_dist1 = average_dist1.type_as(emb1)
average_dist2 = average_dist2.type_as(emb2)
query = emb1[dico[:, 0]]
scores = query.mm(emb2.transpose(0, 1))
scores.mul_(2)
scores.sub_(average_dist1[dico[:, 0]][:, None] + average_dist2[None, :])
results = []
top_matches = scores.topk(10, 1, True)[1]
for k in [1, 5, 10]:
    top_k_matches = top_matches[:, :k]
    _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
    # allow for multiple possible translations
    matching = {}
    for i, src_id in enumerate(dico[:, 0]):
        matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
    # evaluate precision@k
    precision_at_k = 100 * np.mean(list(matching.values()))
    results.append(('precision_at_%i' % k, precision_at_k))
print(results)
