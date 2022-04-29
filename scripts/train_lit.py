import argparse
import torch.nn.parallel
from gazetracker.models.gazetrack_lit import lit_gazetrack_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
# from pytorch_lightning.plugins import DDPPlugin

"""
### Usage:
python train_lit.py 
    --dataset_dir <Path to dataset> 
    --save_dir <Path to save files> 
    --gpus <Number of GPUs to use> 
    --epochs <Number of epochs>
    --comet_name <Name of the experiment on comet.ml>
    --batch_size <Batch size>
    --checkpoint <Path to load checkpoint from to continue training>

### Example: 
python train_lit.py --dataset_dir ../gazetrack/ --save_dir ../Checkpoints/ --gpus 1 --epochs 50 --checkpoint ../Checkpoints/checkpoint.ckpt
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Train GazeTracker')
    parser.add_argument('--dataset_dir', default='../../dataset/', help='Path to converted dataset')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--save_dir', default='./models/', help='Path store checkpoints')
    parser.add_argument('--comet_name', default='gazetrack-lit', help='Path store checkpoints')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--checkpoint', default=None, help='Path to load pre trained weights')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    proj_name = args.comet_name
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, filename='{epoch}-{val_loss:.3f}-{train_loss:.3f}', save_top_k=-1)
    logger = CometLogger(
        api_key="YOUR-API-KEY",
        project_name=proj_name,
    )
    
    model = lit_gazetrack_model(args.dataset_dir, args.save_dir, args.batch_size, logger)
    if args.checkpoint:
        if args.gpus==0:
            w = torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            w = torch.load(args.checkpoint)['state_dict']
        model.load_state_dict(w)
        print("Loaded checkpoint")

    trainer = pl.Trainer(gpus=args.gpus, logger=logger, accelerator="ddp", max_epochs=args.epochs,
                         default_root_dir=args.save_dir, callbacks=[checkpoint_callback],
                         # TODO: we could also use plugins=DDPPlugin(find_unused_parameters=False),
                         progress_bar_refresh_rate=1, auto_lr_find=True, auto_scale_batch_size=True)
    trainer.fit(model)
    print("DONE")
