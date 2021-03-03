import ipdb
from fairseq.data import LanguagePairDataset, IndexedRawLinesDataset
from fairseq.data import data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


@register_task('user_translation')
class UserTranslation(TranslationTask):

    @staticmethod
    def add_args(parser):
        int_inf = 1000000000000
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default=True, type=bool, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default=False, type=bool, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # Limit train, test, dev for downstream task
        # parser.add_argument('--train-limit', type=int, default=int_inf, help='maximum size for downstream train split')
        # parser.add_argument('--valid-limit', type=int, default=int_inf, help='maximum size for downstream dev split')
        # parser.add_argument('--test-limit', type=int, default=int_inf, help='maximum size for downstream test split')
        parser.add_argument('--support-tokens', type=int, default=int_inf, help='maximum size for single task support split')
        parser.add_argument('--query-tokens', type=int, default=int_inf, help='maximum size for single task query split')
        parser.add_argument('--is-curriculum', action='store_true', help='inner curriclum learning option')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, user_data_frame, task_score):
        super().__init__(args, src_dict, tgt_dict)
        self.is_curriculum = args.is_curriculum
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.user_data_frame = user_data_frame
        self.task_score = task_score
        # self.user_data_frame = user_data_frame[user_data_frame.split == args.train_subset].head(args.train_limit)
        # self.user_data_frame = self.user_data_frame.append(user_data_frame[user_data_frame.split == args.valid_subset].head(args.valid_limit))
        # self.user_data_frame = self.user_data_frame.append(user_data_frame[user_data_frame.split == args.test_subset].head(args.test_limit))

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        user_data_frame = kwargs['user_data_frame']
        task_score = kwargs['task_score']
        src_dict = kwargs['src_dict']
        tgt_dict = kwargs['tgt_dict']
        return cls(args=args, src_dict=src_dict, tgt_dict=tgt_dict, user_data_frame=user_data_frame, task_score=task_score)

    # def dataset_buckets(self, split):
        #  split_data = self.datasets[split].src

    def load_dataset(self, split, combine=False, fine_tune=False, bucket=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split not in self.datasets:
            if not fine_tune:
                split_data = self.user_data_frame[self.user_data_frame.task_group == split].reset_index()
            else:
                split_data = self.user_data_frame[self.user_data_frame.task_group.str.contains(split)]
            source_lines = split_data.src
            target_lines = split_data.tgt
            src_dataset = IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict)
            tgt_dataset = IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict)
            print('| {} {} {} examples'.format('', split, len(src_dataset)))
            self.datasets[split] = LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions
            )
            if bucket:
                split = split + '_bucket'
                self.datasets[split] = []
                bucket_info = list(set(split_data.task_group.values))
                bucket_info = sorted(bucket_info, key=lambda x:int(x.split('B')[-1]))
                for bucket_split in bucket_info:
                    bucket_data = split_data[split_data.task_group == bucket_split]
                    source_lines = bucket_data.src
                    target_lines = bucket_data.tgt
                    src_dataset = IndexedRawLinesDataset(lines=source_lines, dictionary=self.src_dict)
                    tgt_dataset = IndexedRawLinesDataset(lines=target_lines, dictionary=self.tgt_dict)
                    print('| {} bucket: {} {} examples'.format('', bucket_split, len(src_dataset)))
                    self.datasets[split].append(LanguagePairDataset(
                        src_dataset, src_dataset.sizes, self.src_dict,
                        tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                        max_source_positions=self.args.max_source_positions,
                        max_target_positions=self.args.max_target_positions,
                        shuffle=not self.is_curriculum
                    )
                    )
