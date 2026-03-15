import torch

def temp_scaling(next_logit, temp):
    prob = torch.softmax(next_logit / temp, dim=-1)
    return prob

def top_p_sampling(probabilities, p):
    if p == 1:
        sort_probabilities = probabilities
        idx = torch.arange(probabilities.shape[-1])
        idx = idx.unsqueeze(0).expand(probabilities.shape[0], -1)
    else:
        sort_probabilities, idx = torch.sort(probabilities, dim=-1, descending=True)
    cdf = torch.cumsum(sort_probabilities, dim=-1)

    mask = cdf > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sort_probabilities = sort_probabilities.masked_fill(mask, 0.0)

    next_token_idx = torch.multinomial(sort_probabilities, 1)
    next_token_idx = idx.gather(-1, next_token_idx)
    return next_token_idx

def decoding(model, input_seq, temp, p, max_num, eos):
    model.eval()
    input_seq = torch.unsqueeze(input_seq, dim=0)
    with torch.no_grad():
        for _ in range(max_num):
             logits = model(input_seq)
             next_logit = logits[:, -1, :]
             probabilities = temp_scaling(next_logit, temp)
             next_token_idx = top_p_sampling(probabilities, p)
             input_seq = torch.cat([input_seq, next_token_idx], dim=-1)
             if next_token_idx.item() == eos:
                 break
    return input_seq
