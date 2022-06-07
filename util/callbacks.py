from tensorflow.keras.callbacks import Callback
from .batch_generator import all_outputs_decode
from .metrics import AP

class EvaluateMeanAP(Callback):
    def __init__(self, network, batch_size, live, train, valid, test, checkpoint_path):
        super(EvaluateMeanAP, self).__init__()
        self.best_valid_mAP = 0
        self.best_test_mAP = 0
        self.network = network
        self.batch_size = batch_size
        self.live = live
        self.train = train
        self.valid = valid
        self.test = test
        self.checkpoint_path = checkpoint_path
    
    def on_epoch_end(self, epoch, logs=None):
        train_y_pred = self.model.predict(self.train['X'], batch_size = self.batch_size, verbose = 0)
        (train_y_pred_main, train_y_pred_subparts) = all_outputs_decode(train_y_pred, self.network, nms_threshold = 0.3)
        train_main_ap     = AP(self.train['y_obj'], train_y_pred_main)
        train_subparts_ap = AP(self.train['y_subparts'], train_y_pred_subparts)
        self.live.log('train_main_ap', train_main_ap.astype(float) )
        self.live.log('train_subparts_ap', train_subparts_ap.astype(float) )

        valid_y_pred = self.model.predict(self.valid['X'], batch_size = self.batch_size, verbose = 0)
        (valid_y_pred_main, valid_y_pred_subparts) = all_outputs_decode(valid_y_pred, self.network, nms_threshold = 0.3)
        val_main_ap     = AP(self.valid['y_obj'], valid_y_pred_main)
        val_subparts_ap = AP(self.valid['y_subparts'], valid_y_pred_subparts)
        self.live.log('val_main_ap', val_main_ap.astype(float) )
        self.live.log('val_subparts_ap', val_subparts_ap.astype(float) )

        test_y_pred = self.model.predict(self.test['X'], batch_size = self.batch_size, verbose = 0)
        (test_y_pred_main, test_y_pred_subparts) = all_outputs_decode(test_y_pred, self.network, nms_threshold = 0.3)
        test_main_ap     = AP(self.test['y_obj'], test_y_pred_main)
        test_subparts_ap = AP(self.test['y_subparts'], test_y_pred_subparts)
        self.live.log('test_main_ap', test_main_ap.astype(float) )
        self.live.log('test_subparts_ap', test_subparts_ap.astype(float) )

        if val_main_ap > self.best_valid_mAP:
            self.best_valid_mAP = val_main_ap
            # self.model.save(f'{RESULTS}/resnet50_{epoch:05d}-{APs[1]:.6f}.hdf5')
            # print('Saving... ')
        
        print(f'train_main_ap: {train_main_ap.astype(float):.6}, train_subparts_ap: {train_subparts_ap.astype(float):.6}, val_main_ap: {val_main_ap.astype(float):.6}, val_subparts_ap: {val_subparts_ap.astype(float):.6}, test_main_ap: {test_main_ap.astype(float):.6}, test_subparts_ap: {test_subparts_ap.astype(float):.6}')
        
        if test_main_ap > self.best_test_mAP:
            filename = f'{self.checkpoint_path}/model.{epoch:06d}.tf'
            print(f'Saving... {filename}')
            self.best_test_mAP = test_main_ap
            self.model.save(filename, save_format='tf')

        self.live.next_step()
