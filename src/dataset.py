def get_char_targets(text,locations):
    char_target = [0] * len(text)
        for location in locations:
            array = location.split()
            start = int(array[0])
            end = int(array[-1])
            for i in range(start,end):
                if text[i] != " ":
                    char_target[i] = 1
    return char_target

def get_targets(input_ids,
                offsets,
                char_targets,
                start_idx):

    targets = [0] * len(input_ids)
    for idx,offset in enumerate(offsets):
        if idx>=start_idx:
            if sum(char_targets[offset[0]:offset[-1]]) > 0:
                targets[i] = 1
    return targets

def get_offsets(temp_offsets):
    offsets = []
    for i in temp_offsets:
        if i[0] == 0:
            offsets.append(i)
        else:
            i = list(i)
            i[0] += 1
            offsets.appen((i[0],i[-1]))
    return offsets


class NBMEDataset:
    def __init__(self,df,tokenizer):
        self.df = df
        self.text = df.pn_history.values
        self.feature_text = df.feture_text.values
        self.annotation = df.annotation.values
        self.location = df.location.values
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        text = self.text[idx]
        featute_text = self.feature_text[idx]
        annotation = self.annotation[idx]
        locations = self.location[idx]

        char_target = get_char_targets(text,locations)

        tok_text = self.tokenizer(
            feature_text,text,
            return_offsets_mapping=True
        )

        input_ids = tok_text["input_ids"]
        token_type_ids = tok_text["token_type_ids"]
        attention_mask = tok_text["attention_mask"]
        temp_offsets = tok_text["offsets_mapping"]

        offsets = get_offsets(temp_offsets)

        start_idx = token_type_ids.index[1]
        targets = get_targets(input_ids,
                             offsets,
                             char_targets,
                             start_idx)
        
        padding_len = self.max - len(input_ids)

        input_ids = input_ids + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        targets = targets + [0] * padding_len

        return {
            "input_ids":torch.tensor(input_ids,dtype=long),
            "token_type_ids":torch.tensor(token_type_ids,dtype=long),
            "attention_mask":torch.tensor(attention_mask,dtype=long),
            "targets":torch.tensor(targets,dtype=long),
            "location":str(locations)
            "offsets":str(offsets)
        }
