import torch
import torch.nn.functional as F

def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):

    net.eval()

    img = torch.from_numpy(full_img)
    img = img.unsqueeze(0)

    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

    probs = probs.squeeze(0)

    full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold
