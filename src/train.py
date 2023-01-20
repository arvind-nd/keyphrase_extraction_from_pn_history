def loss_fn(o1,t1):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    return l1

def train_fn(dataloader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        offset = d["offsets"]
        location = d["location"]
        del offset
        del location

        ids = ids.to(device)
        token_type_ids = token_type_ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        o1 = model(
            ids=ids,
            token_type_ids=token_type_ids,
            mask=mask
        )

        loss = loss_fn(o1,  targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

        ids = ids.cpu().detach().numpy()
        token_type_ids = token_type_ids.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        o1  = o1.cpu().detach().numpy()
        del ids
        del token_type_ids
        del mask
        del targets
        del o1
    return losses.avg


def eval_fn(dataloader, model, device):
    model.eval()
    losses = AverageMeter()
    tk0 = tqdm(dataloader, total=len(dataloader))
    fin_output = []
    fin_targets = []
    fin_offsets = []
    orig_targets_location = []
    fin_targets_location = []
    fin_output_location = []
    TP = []
    FN = []
    FP = []

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        offsets = d["offsets"]
        location = d["location"]


        for i in range(len(location)):
            location[i] = eval(location[i])
            offsets[i] = eval(offsets[i])
        fin_offsets.extend(offsets)
        orig_targets_location.extend(location)

        del offsets
        del location

        ids = ids.to(device)
        token_type_ids = token_type_ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)


        o1 = model(
            ids=ids,
            token_type_ids=token_type_ids,
            mask=mask
        )

        loss = loss_fn(o1, targets)


        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

        ids = ids.cpu().detach().numpy()
        token_type_ids = token_type_ids.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        del ids
        del token_type_ids
        del mask

        threshold = args.threshold

        fin_output.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_targets.append(targets.cpu().detach().numpy())
        del o1
        del targets


    fin_offsets = np.array(fin_offsets)
    orig_targets_location = np.array(orig_targets_location)

    fin_output = np.vstack(fin_output)
    fin_targets = np.vstack(fin_targets)


    for i in range(len(fin_output)):
        output = [1 if i>=threshold else 0 for i in fin_output[i]]
        target = fin_targets[i]
        offset = fin_offsets[i]
        output_location = []
        start = -1
        for j in range(len(output)):
            if output[j]==1 and start==-1:
                start = offset[j][0]
            if output[j]==0 and start!=-1:
                end = offset[j-1][-1]
                output_location.extend([ii for ii in range(start,end)])
                start=-1

        fin_output_location.append(output_location)


    for i in orig_targets_location:
        orig_location = []
        for j in  i:
            m = j.split()
            start = int(m[0])
            end = int(m[-1])
            orig_location.extend([ii for ii in range(start,end)])
        fin_targets_location.append(orig_location)

    for i in range(len(orig_targets_location)):
        ground_truth = fin_targets_location[i]
        prediction = fin_output_location[i]

        tp, fn, fp = confusion_metrics(ground_truth,prediction)
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)

    micro_average_f1_score = micro_average_f1(TP, FN, FP)

    return losses.avg, micro_average_f1_score
