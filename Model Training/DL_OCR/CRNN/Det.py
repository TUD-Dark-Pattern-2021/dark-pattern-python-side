
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from docopt import docopt
from config import common_config as config
from model import CRNN
from ctc_decoder import ctc_decode
from dataset import Synth90kDataset, synth90k_collate_fn

#create crnn model to create
def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad(): 
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu' #use gpu if availible

            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds

#give the result
def show_result(paths, preds):
    print('\n---result---')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():
    arguments = docopt(__doc__)
    reload_checkpoint = arguments['-m']
    images = arguments['IMAGE']
    decode_method = arguments['-d']
    batch_size = int(arguments['-s'])
    beam_size = int(arguments['-b'])
    
# read img config
    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    predict_dataset = Synth90kDataset(paths=images,
                                      img_height=img_height, img_width=img_width)

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,  # divide the space
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
