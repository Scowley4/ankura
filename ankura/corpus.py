"""Provides access to some standard downloadable datasets.

The available datasets (and corresponding import functions) include:
    * bible
    * newsgroups
    * amazon
    * tripadvisor
    * yelp
    * nyt
These imports depend on two module variables which can be mutated to change the
download behavior of these imports. Downloaded and pickled data will be stored
in the path given by `download_dir`, and data will be downloaded from
`base_url`. By default, `download_dir` will be '$HOME/.ankura' while base_url
will point at a GitHub repo designed for use with
"""
import json
import functools
import itertools
import os
import string
import urllib.request
import re

from . import pipeline
import posixpath

download_dir = os.path.join(os.getenv('HOME'), '.ankura')

def _path(name, *opts):
    if opts:
        name, dot, ext = name.partition('.')
        opts =  '_'.join('-' if opt is None else str(int(opt)) for opt in opts)
        name = '{}_{}{}{}'.format(name, opts, dot, ext)
    return os.path.join(download_dir, name)


base_url = 'https://github.com/jefflund/data/raw/data2'

def _url(name):
    return posixpath.join(base_url, name)


def _ensure_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass


def _ensure_download(name):
    path = _path(name)
    if not os.path.isfile(path):
        _ensure_dir(path)
        urllib.request.urlretrieve(_url(name), path)


def open_download(name, mode='r'):
    """Gets a file object for the given name, downloading the data to download
    dir from base_url if needed. By default the files are opened in read mode.
    If used as part of an inputer, the mode should likely be changed to binary
    mode. For a list of useable names with the default base_url, see
    download_inputer.
    """
    _ensure_download(name)
    return open(_path(name), mode)


def download_inputer(*names):
    """Generates file objects for the given names, downloading the data to
    download_dir from base_url if needed. The expected names reflect the
    filenames in the default base_url and are used throughout this module. In
    otherwords, if base_url is changed, it may break the import functions in
    the module which rely on download_inputer.
    """
    @functools.wraps(download_inputer)
    def _inputer():
        for name in names:
            yield open_download(name, mode='rb')
    return _inputer


def bible(version='esv', remove_stopwords=True, remove_empty=False, use_stemmer=False):
    """Gets a Corpus containing the the Holy Bible with cross references.

    The translation used can be selected with version (default: 'esv').
    Available options include 'esv' and 'kjv'.
    If remove_stopwords is True (default: True), then stopwords, both modern
    and Jacobean will be pruned from the corpus vocabulary.
    If remove_empty is True (default: False), then words which appear in 0 or 1
    documents will be removed, and any empty documents will be removed.
    If use_stemmer is True (default: False), then each token will be stemmed
    using the Porter stemming algorithm.
    """
    tokenizer = pipeline.translate_tokenizer(
        pipeline.split_tokenizer(string.whitespace + '—'),
    )
    if remove_stopwords:
        tokenizer = pipeline.stopword_tokenizer(
            tokenizer,
            open_download('stopwords/nltk.txt'),
        )
    if use_stemmer:
        tokenizer = pipeline.stemming_tokenizer(tokenizer)

    def _xref_labeler(name):
        return pipeline.list_labeler(
            open_download('bible/xref-{}.txt'.format(name)),
            'xref-{}'.format(name),
        )

    p = pipeline.Pipeline(
        download_inputer('bible/{}.txt'.format(version)),
        pipeline.line_extractor('\t'),
        tokenizer,
        pipeline.composite_labeler(
            pipeline.title_labeler('verse'),
            _xref_labeler('tske'),
            _xref_labeler('obib'),
            *[_xref_labeler('obib{}'.format(i)) for i in range(11)],
        ),
        pipeline.keep_filterer(),
        pipeline.composite_informer(
            pipeline.title_informer('verses', 'verse'),
            pipeline.kwargs_informer(
                name='Holy Bible',
                version=version,
                remove_stopwords=remove_stopwords,
                remove_empty=remove_empty,
                use_stemmer=use_stemmer,
            ),
        )
    )

    if remove_empty:
        p.tokenizer = pipeline.frequency_tokenizer(p, 2)
    bible = p.run(_path('bible_{}.pickle'.format(version),
        remove_stopwords,
        remove_empty,
        use_stemmer,
    ))
    if remove_empty: # frequency_tokenizer may remove last word of a doc
        bible = pipeline.select_docs(bible, lambda d: d.tokens)

    return bible


def newsgroups(rare_threshold=100, common_threshold=2000):
    """Gets a Corpus containing roughly 20,000 usenet postings from 20
    different newsgroups in the early 1990's.

    The rare_threshold (default: 100) is the minimum number of documents a word
    must appear in to be retained.
    The common_threshold (default: 2000) is the maximum number of documents a
    word can appear in to be retained.
    Both options can be set to None to disable filtering.
    """
    coarse_mapping = {
        'comp.graphics': 'comp',
        'comp.os.ms-windows.misc': 'comp',
        'comp.sys.ibm.pc.hardware': 'comp',
        'comp.sys.mac.hardware': 'comp',
        'comp.windows.x': 'comp',
        'rec.autos': 'rec',
        'rec.motorcycles': 'rec',
        'rec.sport.baseball': 'rec',
        'rec.sport.hockey': 'rec',
        'sci.crypt': 'sci',
        'sci.electronics': 'sci',
        'sci.med': 'sci',
        'sci.space': 'sci',
        'misc.forsale': 'misc',
        'talk.politics.misc': 'politics',
        'talk.politics.guns': 'politics',
        'talk.politics.mideast': 'politics',
        'talk.religion.misc' : 'religion',
        'alt.atheism' : 'religion',
        'soc.religion.christian' : 'religion',
    }

    p = pipeline.Pipeline(
        download_inputer('newsgroups/newsgroups.tar.gz'),
        pipeline.targz_extractor(
            pipeline.skip_extractor(errors='replace'),
        ),
        pipeline.remove_tokenizer(
            pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                itertools.chain(open_download('stopwords/english.txt'),
                                open_download('stopwords/newsgroups.txt'))
            ),
            r'^(.{0,2}|.{15,})$', # remove any token t for which 2<len(t)<=15
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.dir_labeler('newsgroup'),
            pipeline.transform_labeler(
                pipeline.dir_labeler('coarse_newsgroup'),
                coarse_mapping.get,
            ),
        ),
        pipeline.length_filterer(),
        pipeline.kwargs_informer(name='newsgroups'),
    )
    if rare_threshold or common_threshold:
        p.tokenizer = pipeline.frequency_tokenizer(p,
            rare_threshold,
            common_threshold,
        )
    return p.run(_path('newsgroups.pickle',
        rare_threshold,
        common_threshold,
    ))


def amazon(rare_threshold=50):
    """Gets a Corpus containing roughly 40,000 Amazon product reviews, with
    star ratings.

    The rare_threshold (default: 50) is the mimimum number of documents word
    must appear in to be retained. It may be set to None to disable filtering.
    """
    p = pipeline.Pipeline(
        download_inputer('amazon/amazon.txt'),
        pipeline.line_extractor('\t'),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('amazon/amazon.stars'),
                'rating',
            ),
            pipeline.transform_labeler(
                pipeline.float_labeler(
                    open_download('amazon/amazon.stars'),
                    'binary_rating',
                ),
                lambda r: r >= 5,
            ),
        ),
        pipeline.length_filterer(),
        pipeline.kwargs_informer(name='amazon'),
    )
    if rare_threshold:
        p.tokenizer = pipeline.frequency_tokenizer(p, rare_threshold)
    return p.run(_path('amazon.pickle', rare_threshold))


def amazon_large():
    class popiter(list):
        def __iter__(self):
            while self:
                yield self.pop()
    label_stream = popiter()

    def hingidy_jingidies(docfile, value_key='reviewText', label_key='overall'):
        for i, line in enumerate(docfile):
            if i == 1000000:
                break
            line = json.loads(line.decode('utf-8'))
            label_stream.append((str(i), line[label_key]))
            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        pipeline.file_inputer('/users/data/amazon_large/item_dedup.json.gz'),
        pipeline.gzip_extractor(hingidy_jingidies),
        pipeline.default_tokenizer(),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 200, 100000)
    return p.run(_path('amazon_large.pickle'), _path('amazon_large.docs'))


def tripadvisor():
    """Gets a Corpus containing roughly 240,000 TripAdvisor hotel reviews, with
    ratings for various aspects of the hotel.
    """
    labeler = {}

    def _extractor(docfile):
        hotel_id = docfile.name[6:-4]
        reviews = docfile.read().decode('utf-8').split('\r\n\r\n')[1:]
        for i, review in enumerate(reviews):
            review = review.split('\r\n')
            if len(review) != 13:
                continue
            review_id = '{}_{}'.format(hotel_id, i)
            labeler[review_id] = dict(line[1:].split('>') for line in review[5:])
            yield pipeline.Text(review_id, review[1][9:])

    p = pipeline.Pipeline(
        download_inputer('tripadvisor/tripadvisor.tar.gz'),
        pipeline.targz_extractor(_extractor),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('Id'),
            labeler.pop,
        ),
        pipeline.length_filterer(),
        pipeline.kwargs_informer(name='tripadvisor'),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 200)
    return p.run(_path('tripadvisor.pickle'))


def yelp():
    """Gets a Corpus containing roughly 25,000 restaurant reviews, along with
    ratings.
    """
    p = pipeline.Pipeline(
        download_inputer('yelp/yelp.txt'),
        pipeline.line_extractor('\t'),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('yelp/yelp.response'),
                'rating',
            ),
            pipeline.transform_labeler(
                pipeline.float_labeler(
                    open_download('yelp/yelp.response'),
                    'binary_rating',
                ),
                lambda r: r >= 5,
            ),
        ),
        pipeline.length_filterer(),
        pipeline.kwargs_informer(name='yelp'),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('yelp.pickle'))


def nyt(rare_threshold=150):
    """Gets a Corpus containing roughly 40,000 news stories from 2004 published
    by the New York Times.

    The rare_threshold (default: 150) is the mimimum number of documents word
    must appear in to be retained. It may be set to None to disable filtering.
    """
    p = pipeline.Pipeline(
        download_inputer('nyt/nyt.tar.gz'),
        pipeline.targz_extractor(pipeline.whole_extractor()),
        pipeline.remove_tokenizer(
            pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                itertools.chain(open_download('stopwords/english.txt'),
                                open_download('stopwords/newsgroups.txt'))
            ),
            r'^(.{0,2}|.{15,})$', # remove any token t for which 2<len(t)<=15
        ),
        pipeline.title_labeler('id'),
        pipeline.length_filterer(),
        pipeline.kwargs_informer(name='nyt'),
    )
    if rare_threshold:
        p.tokenizer = pipeline.frequency_tokenizer(p, rare_threshold)
    return p.run(_path('nyt.pickle', rare_threshold))


def beowulf(remove_stopwords=True, use_stemmer=False):
    """Imports the Heaney translation of Beowulf.
    """
    def poetry_extractor(docfile):
        for lineno, line in enumerate(docfile):
            yield pipeline.Text(lineno, line.decode('utf-8', 'strict').strip())
    def keepalpha_tokenizer(base_tokenizer):
        pattern = re.compile('[\W]+')
        modify = lambda t: pattern.sub('', t.lower())
        def _tokenizer(data):
            tokens = [pipeline.TokenLoc(modify(t.token), t.loc) for t in base_tokenizer(data)]
            return [t for t in tokens if t.token]
        return _tokenizer

    tokenizer = keepalpha_tokenizer(pipeline.split_tokenizer())
    if remove_stopwords:
        tokenizer = pipeline.stopword_tokenizer(
            tokenizer,
            open_download('stopwords/nltk.txt'),
        )
    if use_stemmer:
        tokenizer = pipeline.stemming_tokenizer(tokenizer)

    p = pipeline.Pipeline(
        download_inputer('beowulf/heaney.txt'),
        poetry_extractor,
        tokenizer,
        pipeline.title_labeler('lineno'),
        pipeline.keep_filterer(),
        pipeline.kwargs_informer(
            name='Beowulf',
            remove_stopwords=remove_stopwords,
            use_stemmer=use_stemmer,
        ),
    )

    return p.run(_path('beowulf.pickle',
        remove_stopwords,
        use_stemmer,
    ))

def artofwar(remove_stopwords=True, use_stemmer=False):
    """Imports the sunzisaid translation of Sunzi's Art of War (孙子兵法)
    """

    def replace_pretokenizer(base_tokenizer, find='---', repl=' '):
        def _tokenizer(data):
            return base_tokenizer(data.replace(find, repl))
        return _tokenizer
:wa

    tokenizer = replace_pretokenizer(pipeline.default_tokenizer())

    def artofwar_extractor(docfile):
        chapter = None
        num_re = re.compile(r'\d+')

        for line in docfile:
            line = line.decode('utf-8', 'strict')

            # Skip blank lines
            if not line.strip():
                continue

            # Extract the chapter
            if line.startswith('CHAPTER'):
                chapter = int(num_re.findall(line)[0])
                continue

            # Skip lines before the first chapter
            if chapter is None:
                continue

            verse, text = line.split('.', 1)
            yield pipeline.Text(f'{chapter}:{verse}', text.strip())

    # inputer, extractor, tokenizer, labler, filterer, informer=None
    p = pipeline.Pipeline(
          download_inputer('artofwar/sunzisaid.txt'),
          artofwar_extractor,
          tokenizer,
          pipeline.title_labeler('verse'),
          pipeline.keep_filterer(),
          pipeline.kwargs_informer(
              name='ArtOfWar',
              remove_stopwords=remove_stopwords,
              use_stemmer=use_stemmer,
          ),
    )


    return p.run(_path('artofwar.pickle',
        remove_stopwords,
        use_stemmer,
    ))
