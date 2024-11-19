import os
import cv2.data
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import cv2

class GarbageDetector():
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.camera = cv2.VideoCapture(0)

    def run(self):
        data_dir  = 'data'

        classes = os.listdir(data_dir)
        print(classes)

        transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        dataset = ImageFolder(data_dir, transform = transformations)

        def get_default_device():
            """Pick GPU if available, else CPU"""
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
            
        def to_device(data, device):
            """Move tensor(s) to chosen device"""
            if isinstance(data, (list,tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)

        device = get_default_device()

        def predict_image(img, model):
            # Convert to a batch of 1
            xb = to_device(img.unsqueeze(0), device)
            # Get predictions from model
            yb = model(xb)
            # Pick index with highest probability
            prob, preds  = torch.max(yb, dim=1)
            # Retrieve the class label
            return dataset.classes[preds[0].item()]

        def predict_external_image(image_name):
            image = Image.fromarray(image_name).convert('RGB')
            example_image = transformations(image)
            response = predict_image(example_image, self.model)

            return response

        counter = 0

        while True:
            ret, frame = self.camera.read()

            if ret:
                if counter % 30 == 0:
                    try:
                        garbage_type = predict_external_image(frame)
                        frame = cv2.putText(frame, garbage_type, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    except ValueError:
                        pass

                counter += 0

                cv2.imshow('video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    garbage_detector = GarbageDetector('wall-e.pt')
    garbage_detector.run()
