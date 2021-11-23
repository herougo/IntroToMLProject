from torch.utils.tensorboard import SummaryWriter
import numpy as np
import PIL.Image
import warnings

ORIGINAL_DATAFORMAT = {
    'C': 2,
    'H': 0,
    'W': 1
}

class TensorboardWrapper:
    ''' Notes:
    - to get values on the same plot, use the same tag
    
    '''
    
    def __init__(self, log_dir=None, **kwargs):
        self.writer = SummaryWriter(log_dir, **kwargs)
    
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        '''
        e.g. w.add_scalar('model1/loss', 0.0, i)
        scalar_value: (float or string/blobname): Value to save
        global_step: 
        '''
        if scalar_value is None:
            warnings.warn('TensorboardWrapper.add_scalar None value. Doing nothing.')
            return

        self.writer.add_scalar(tag, scalar_value, global_step=global_step, 
                               walltime=walltime)
    
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        '''
        Example:
        writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                       'xcosx':i*np.cos(i/r),
                                       'tanx': np.tan(i/r)}, i)
        '''
        if tag_scalar_dict is None:
            warnings.warn('TensorboardWrapper.add_scalars None value. Doing nothing.')
            return
        
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step=global_step, 
                               walltime=walltime)
        
    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None,
                      max_bins=None):
        # values (torch.Tensor, numpy.array, or string/blobname): Values to build histogram
        if values is None:
            warnings.warn('TensorboardWrapper.add_histogram None value. Doing nothing.')
            return
        
        self.writer.add_histogram(tag, values, global_step=global_step, bins=bins, 
                                  walltime=walltime, max_bins=max_bins)
        
    def add_image(self, tag, img, global_step=None, walltime=None, dataformats='HWC'):
        # img: PIL image or tensor (torch.Tensor, numpy.array, or string/blobname): Image data
        # dataformats: shape of numpy img (or desired PIL conversion)
        if img is None:
            warnings.warn('TensorboardWrapper.add_image None value. Doing nothing.')
            return
        
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img).astype('float32') / 255 # HWC
            new_axis_order = [ORIGINAL_DATAFORMAT[c] for c in dataformats]
            img = np.moveaxis(img, new_axis_order, [0, 1, 2])
        self.writer.add_image(tag, img, global_step=global_step, walltime=walltime, 
                              dataformats=dataformats)

    
    def flush(self):
        self.writer.flush()
    
    def close(self):
        self.writer.close()