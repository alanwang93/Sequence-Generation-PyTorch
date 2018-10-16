
import s2s.config as config
from s2s.utils.dataloader import build_dataloaders 
from s2s.utils.vocab import Vocab

cf = getattr(config, 'Test')()
dc = cf.dataset
src_vocab = Vocab(path=dc.path, max_vocab=dc.max_src_vocab, \
        min_freq=dc.min_freq, prefix='src', unk_token=dc.unk_token)
tgt_vocab = Vocab(path=dc.path, max_vocab=dc.max_tgt_vocab, \
        min_freq=dc.min_freq, prefix='tgt', unk_token=dc.unk_token)

src_vocab.build(rebuild=False)
tgt_vocab.build(rebuild=False)
train, dev, test = build_dataloaders(cf, src_vocab, tgt_vocab)

iter_train = iter(train)
for i, data in enumerate(iter_train):
    print(i, data)


