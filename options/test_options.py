from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', default='', type=str, help='the results dir, default is expr_dir/results  ')
        self.parser.add_argument('--n_samples', type=int, default=5, help='#samples for multimodal')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False
