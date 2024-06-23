import wandb
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet
import os
from dotenv import load_dotenv



MODELS_DIR = 'models'
MODEL_FILE_NAME = 'model.pth'

CATEGORIES=['freshapple', 'freshbanana', 'freshorange', 
            'rottenapple', 'rottenbanana', 'rottenorange']

load_dotenv()
os.getenv('WANDB_API_KEY') # this one gets the API key into os.environ

# this is what comes out of wandb

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, 'Please enter the wandb API key'
    wandb_org = os.environ.get('WANDB_ORG')
    assert 'WANDB_ORG' in os.environ
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')
    
    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    wandb.login()
    #this creates an artifact object
    artifact = wandb.Api().artifact(artifact_path, type='model')
    #this calls the download method from the artifact object
    artifact.download(root=MODELS_DIR)




#download_artifact()



