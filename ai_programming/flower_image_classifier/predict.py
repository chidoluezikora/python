import argparse, torch, json
from PIL import Image
from torchvision import transforms



def parse_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input',
        action = 'store',
        type = str,
        help = 'full path to a flower image file'
    )
    parser.add_argument(
        'checkpoint',
        action = 'store',
        type = str,
        help = 'path to the checkpoint file'
    )
    parser.add_argument(
        '--top_k',
        type = int,
        default = 5,
        help = 'top k most likely flowers from classifier'
    )
    parser.add_argument(
        '--category_names',
        type = str,
        default = 'cat_to_name.json',
        help = 'path to json file with category names'
    )
    parser.add_argument(
        '--gpu',
        action = 'store_true',
        default = False,
        help = 'pass argument to use gpu'
    )

    args = parser.parse_args()
    
    return args


in_arg = parse_input_args()
input = in_arg.input
checkpoint = in_arg.checkpoint
top_k = in_arg.top_k
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if in_arg.category_names:
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)


def load_checkpoint():
    data = torch.load(checkpoint)
    optimizer_dict = data['optimizer_state_dict']
    model = data['model']
    model.classifier = data['classifier']
    model.class_to_idx  = data['class_to_dict']
    model.load_state_dict(data['model_state_dict'])
    
    return model, optimizer_dict


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    array = processor(image)
    array.unsqueeze_(0)
    
    return array.numpy()


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    flower_image = process_image(Image.open(image_path))
    image = torch.from_numpy(flower_image)
    model, image = model.to(device), image.to(device)
    with torch.no_grad():
        output = model(image)
        pb = torch.exp(output)
    top_pb, top_class = pb.topk(topk)
    top_pb = top_pb.tolist()[0]
    top_class = top_class.tolist()[0]
    data = {val: key for key, val in model.class_to_idx.items()}
    top_flow = []
    for i in top_class:
        j = "{}".format(data.get(i))
        top_flow.append(cat_to_name.get(j))

    return top_pb, top_flow


model, optimizer_dict = load_checkpoint()
ps, flowers = predict(input, model, top_k)


print("The flower is classified as {} with a likelihood of {:.3f}%!\n\n\n".format(flowers[0].capitalize(), ps[0]*100))
print("Other likely flowers are:\n\n")

for i in range(1, top_k):
    print("{} with {}%\n".format(flowers[i].capitalze(), ps[i]*100))