from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=30, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')
        # learning rate
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # lambda parameters
        self.parser.add_argument('--lambda_cyc', type=float, default=10.0, help='weight for cycle consistency')
        self.parser.add_argument('--lambda_ide', type=float, default=0, help='weight for identity consistency')
        self.parser.add_argument('--lambda_c', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
        self.parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        self.isTrain = True
