import os
from typing import Iterable
from PIL import Image

from species_segmentation import DatasetConfig


_TAR_PATH = "/mmfs1/gscratch/krishna/sgeng/datasets/iwildcam.tar"
_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
_IMAGES_DIR = os.path.join(_DATA_ROOT, "train")
_METADATA_CSV = os.path.join(_DATA_ROOT, "metadata.csv")
_CATEGORIES_CSV = os.path.join(_DATA_ROOT, "categories.csv")


def _ensure_data():
    if not os.path.exists(_METADATA_CSV):
        raise RuntimeError(
            f"iWildCam data not found at {_DATA_ROOT}.\n"
            "Run the extraction script first:\n"
            "  sbatch extract_iwildcam.sh\n"
            "or manually:\n"
            f"  mkdir -p {_DATA_ROOT} && cd {_DATA_ROOT} && "
            f"tar -xf {_TAR_PATH} --strip-components=1"
        )


def _iwildcam_stream(
    num_samples: int,
    shard_id: int = 0,
    total_shards: int = 1,
    categories=None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> Iterable:
    import pandas as pd

    _ensure_data()

    cats_df = pd.read_csv(_CATEGORIES_CSV)
    label_to_name = dict(zip(cats_df["y"], cats_df["name"]))

    meta = pd.read_csv(_METADATA_CSV)
    # skip empty class (y == 0)
    meta = meta[meta["y"] != 0].reset_index(drop=True)

    shard_df = meta.iloc[shard_id::total_shards].reset_index(drop=True)
    if shuffle:
        shard_df = shard_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    yielded = 0
    for _, row in shard_df.iterrows():
        if yielded >= num_samples:
            break
        species = label_to_name.get(row["y"])
        if species not in _CLASS_MAPPING:
            continue
        if categories is not None and species not in categories:
            continue
        img_path = os.path.join(_IMAGES_DIR, row["filename"])
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  Skipping {img_path}: {exc}")
            continue
        yield {
            "image":    pil_img,
            "species":  species,
            "location": row.get("location"),
            "split":    row.get("split"),
        }
        yielded += 1


# Maps iWildCam scientific names to visual concept terms for SAM segmentation.
_CLASS_MAPPING = {
    # --- Neotropical mammals ---
    "tayassu pecari":           ["pig", "mammal"],
    "dasyprocta punctata":      ["rodent", "mammal"],
    "cuniculus paca":           ["rodent", "mammal"],
    "puma concolor":            ["puma", "mountain lion", "big cat", "mammal"],
    "tapirus terrestris":       ["mammal"],
    "pecari tajacu":            ["pig", "mammal"],
    "mazama americana":         ["deer", "mammal"],
    "leopardus pardalis":       ["cat", "mammal"],
    "geotrygon montana":        ["bird"],
    "nasua nasua":              ["raccoon", "mammal"],
    "dasypus novemcinctus":     ["armadillo", "mammal"],
    "eira barbara":             ["weasel", "mammal"],
    "didelphis marsupialis":    ["opossum", "mammal"],
    "procyon cancrivorus":      ["raccoon", "mammal"],
    "panthera onca":            ["jaguar", "big cat", "mammal"],
    "myrmecophaga tridactyla":  ["anteater", "mammal"],
    "tinamus major":            ["bird"],
    "sylvilagus brasiliensis":  ["rabbit", "mammal"],
    "puma yagouaroundi":        ["cat", "mammal"],
    "puma yagoroundi":          ["cat", "mammal"],
    "leopardus wiedii":         ["cat", "mammal"],
    "mazama gouazoubira":       ["deer", "mammal"],
    "mazama sp":                ["deer", "mammal"],
    "mazama temama":            ["deer", "mammal"],
    "mazama  temama":           ["deer", "mammal"],
    "mazama pandora":           ["deer", "mammal"],
    "philander opossum":        ["opossum", "mammal"],
    "didelphis sp":             ["opossum", "mammal"],
    "nasua narica":             ["raccoon", "mammal"],
    "tamandua mexicana":        ["anteater", "mammal"],
    "dasyprocta fuliginosa":    ["rodent", "mammal"],
    "myoprocta pratti":         ["rodent", "mammal"],
    "proechimys sp":            ["rodent", "mammal"],
    "agouti paca":              ["rodent", "mammal"],
    "conepatus semistriatus":   ["skunk", "mammal"],
    "procyon lotor":            ["raccoon", "mammal"],
    "odocoileus virginianus":   ["deer", "mammal"],
    "urocyon cinereoargenteus": ["fox", "mammal"],
    "cerdocyon thous":          ["fox", "mammal"],
    "canis latrans":            ["coyote", "dog", "mammal"],
    "mustela lutreolina":       ["weasel", "mammal"],
    "peromyscus sp":            ["mouse", "rodent", "mammal"],
    "sciurus sp":               ["squirrel", "mammal"],
    "tapirus bairdii":          ["tapir", "mammal"],
    # --- Neotropical birds ---
    "geotrygon sp":             ["dove", "bird"],
    "penelope purpurascens":    ["bird"],
    "meleagris ocellata":       ["turkey", "bird"],
    "crax rubra":               ["bird"],
    "ortalis vetula":           ["bird"],
    "nothocrax urumutum":       ["bird"],
    "psophia crepitans":        ["bird"],
    "momotus momota":           ["bird"],
    "leptotila plumbeiceps":    ["dove", "bird"],
    "claravis pretiosa":        ["dove", "bird"],
    "unknown dove":             ["dove", "bird"],
    "aramides cajanea":         ["bird"],
    "aramus guarauna":          ["bird"],
    "tigrisoma mexicanum":      ["heron", "bird"],
    "aguila sp":                ["eagle", "bird"],
    "phaetornis sp":            ["hummingbird", "bird"],
    "brotogeris sp":            ["parakeet", "bird"],
    "ave desconocida":          ["bird"],
    # --- Neotropical reptiles ---
    "paleosuchus sp":           ["caiman", "crocodile"],
    # --- African mammals ---
    "capra aegagrus":           ["goat", "mammal"],
    "bos taurus":               ["cow", "cattle", "mammal"],
    "ovis aries":               ["sheep", "mammal"],
    "canis lupus":              ["wolf", "dog", "mammal"],
    "lepus saxatilis":          ["hare", "rabbit", "mammal"],
    "papio anubis":             ["baboon", "mammal"],
    "genetta genetta":          ["genet", "mammal"],
    "tragelaphus scriptus":     ["antelope", "mammal"],
    "equus africanus":          ["donkey", "mammal"],
    "herpestes sanguineus":     ["mongoose", "mammal"],
    "loxodonta africana":       ["elephant", "mammal"],
    "cricetomys gambianus":     ["rat", "rodent", "mammal"],
    "raphicerus campestris":    ["antelope", "mammal"],
    "hyaena hyaena":            ["hyena", "mammal"],
    "aepyceros melampus":       ["antelope", "mammal"],
    "crocuta crocuta":          ["hyena", "mammal"],
    "caracal caracal":          ["cat", "mammal"],
    "equus ferus":              ["horse", "mammal"],
    "panthera leo":             ["lion", "big cat", "mammal"],
    "tragelaphus oryx":         ["antelope", "mammal"],
    "kobus ellipsiprymnus":     ["antelope", "mammal"],
    "phacochoerus africanus":   ["warthog", "pig", "mammal"],
    "panthera pardus":          ["leopard", "big cat", "mammal"],
    "ichneumia albicauda":      ["mongoose", "mammal"],
    "canis mesomelas":          ["jackal", "dog", "mammal"],
    "canis adustus":            ["jackal", "dog", "mammal"],
    "canis familiaris":         ["dog", "mammal"],
    "syncerus caffer":          ["buffalo", "mammal"],
    "equus quagga":             ["zebra", "mammal"],
    "giraffa camelopardalis":   ["giraffe", "mammal"],
    "alcelaphus buselaphus":    ["antelope", "mammal"],
    "chlorocebus pygerythrus":  ["monkey", "mammal"],
    "madoqua guentheri":        ["antelope", "mammal"],
    "potamochoerus larvatus":   ["pig", "mammal"],
    "nanger granti":            ["gazelle", "mammal"],
    "eudorcas thomsonii":       ["gazelle", "mammal"],
    "orycteropus afer":         ["aardvark", "mammal"],
    "acinonyx jubatus":         ["cheetah", "big cat", "mammal"],
    "felis silvestris":         ["cat", "mammal"],
    "oryx beisa":               ["antelope", "mammal"],
    "helogale parvula":         ["mongoose", "mammal"],
    "lycaon pictus":            ["wild dog", "mammal"],
    "procavia capensis":        ["rock hyrax", "mammal"],
    "ictonyx striatus":         ["weasel", "mammal"],
    "otocyon megalotis":        ["fox", "mammal"],
    "equus grevyi":             ["zebra", "mammal"],
    "proteles cristata":        ["hyena", "mammal"],
    "leptailurus serval":       ["cat", "mammal"],
    "tragelaphus strepsiceros": ["antelope", "mammal"],
    "hippopotamus amphibius":   ["hippopotamus", "mammal"],
    "xerus rutilus":            ["squirrel", "mammal"],
    "camelus dromedarius":      ["camel", "mammal"],
    "cephalophus nigrifrons":   ["antelope", "mammal"],
    "cephalophus silvicultor":  ["antelope", "mammal"],
    "atherurus africanus":      ["porcupine", "mammal"],
    "hystrix cristata":         ["porcupine", "mammal"],
    "pan troglodytes":          ["chimpanzee", "mammal"],
    "cercopithecus mitis":      ["monkey", "mammal"],
    "cercopithecus lhoesti":    ["monkey", "mammal"],
    "funisciurus carruthersi":  ["squirrel", "mammal"],
    "protoxerus stangeri":      ["squirrel", "mammal"],
    "paraxerus boehmi":         ["squirrel", "mammal"],
    "genetta servalina":        ["genet", "mammal"],
    "genetta tigrina":          ["genet", "mammal"],
    "nandinia binotata":        ["civet", "mammal"],
    "alopochen aegyptiaca":     ["goose", "bird"],
    # --- African birds ---
    "turtur calcospilos":       ["dove", "bird"],
    "struthio camelus":         ["ostrich", "bird"],
    "eupodotis senegalensis":   ["bustard", "bird"],
    "lophotis gindiana":        ["bustard", "bird"],
    "ardeotis kori":            ["bustard", "bird"],
    "lissotis melanogaster":    ["bustard", "bird"],
    "burhinus capensis":        ["bird"],
    "acryllium vulturinum":     ["guineafowl", "bird"],
    "streptopilia senegalensis": ["dove", "bird"],
    "streptopelia lugens":      ["dove", "bird"],
    "unknown bird":             ["bird"],
    "motacilla flava":          ["wagtail", "bird"],
    "andropadus latirostris":   ["bird"],
    "andropadus virens":        ["bird"],
    "melocichla mentalis":      ["warbler", "bird"],
    "dioptrornis fischeri":     ["flycatcher", "bird"],
    "musophaga rossae":         ["bird"],
    "turtur tympanistria":      ["dove", "bird"],
    "eurocephalus rueppelli":   ["shrike", "bird"],
    "francolinus africanus":    ["bird"],
    "francolinus nobilis":      ["bird"],
    "mesopicos griseocephalus": ["woodpecker", "bird"],
    "turdus olivaceus":         ["thrush", "bird"],
    # --- African small mammals ---
    "thamnomys venustus":       ["rodent", "mammal"],
    "oenomys hypoxanthus":      ["rat", "rodent", "mammal"],
    "hybomys univittatus":      ["rodent", "mammal"],
    "colomys goslingi":         ["mouse", "rodent", "mammal"],
    "hylomyscus stella":        ["mouse", "rodent", "mammal"],
    "praomys tullbergi":        ["rat", "rodent", "mammal"],
    "malacomys longipes":       ["mouse", "rodent", "mammal"],
    "deomys ferrugineus":       ["mouse", "rodent", "mammal"],
    "mus minutoides":           ["mouse", "rodent", "mammal"],
    "unknown bat":              ["bat", "mammal"],
    # --- Southeast Asian mammals ---
    "argusianus argus":         ["pheasant", "bird"],
    "prionailurus bengalensis":  ["cat", "mammal"],
    "hemigalus derbyanus":      ["mammal"],
    "muntiacus muntjak":        ["deer", "mammal"],
    "sus scrofa":               ["wild boar", "pig", "mammal"],
    "helarctos malayanus":      ["sun bear", "bear", "mammal"],
    "rusa unicolor":            ["deer", "mammal"],
    "hystrix brachyura":        ["porcupine", "mammal"],
    "pardofelis temminckii":    ["cat", "mammal"],
    "panthera tigris":          ["tiger", "big cat", "mammal"],
    "lariscus insignis":        ["squirrel", "mammal"],
    "chalcophaps indica":       ["dove", "bird"],
    "paguma larvata":           ["civet", "mammal"],
    "pardofelis marmorata":     ["cat", "mammal"],
    "cuon alpinus":             ["wild dog", "mammal"],
    "varanus salvator":         ["monitor lizard", "lizard"],
    "martes flavigula":         ["marten", "mammal"],
    "prionodon linsang":        ["civet", "mammal"],
    "rollulus rouloul":         ["partridge", "bird"],
    "lophura inornata":         ["pheasant", "bird"],
    "lophura sp":               ["pheasant", "bird"],
    "lophura erythrophthalma":  ["pheasant", "bird"],
    "polyplectron chalcurum":   ["pheasant", "bird"],
    "manis javanica":           ["pangolin", "mammal"],
    "capricornis sumatraensis": ["goat", "mammal"],
    "macaca sp":                ["monkey", "mammal"],
    "callosciurus notatus":     ["squirrel", "mammal"],
    "presbytis thomasi":        ["monkey", "mammal"],
    "neofelis diardi":          ["cat", "mammal"],
    "arctonyx hoevenii":        ["badger", "mammal"],
    "tragulus sp":              ["deer", "mammal"],
    "dendrocitta occipitalis":  ["bird"],
    "niltava sumatrana":        ["flycatcher", "bird"],
    "leiothrix argentauris":    ["bird"],
    "myiophoneus melanurus":    ["thrush", "bird"],
    "myiophoneus glaucinus":    ["thrush", "bird"],
    "myiophoneus caeruleus":    ["thrush", "bird"],
    "arborophila rubrirostris": ["partridge", "bird"],
    "erithacus cyane":          ["robin", "bird"],
    "spilornis cheela":         ["eagle", "bird"],
    "herpestes semitorquatus":  ["mongoose", "mammal"],
    "collocalia linchi":        ["swift", "bird"],
}


IWILDCAM_CONFIG = DatasetConfig(
    name="iWildCam",
    load_fn=_iwildcam_stream,
    get_image=lambda s: s["image"],
    get_class=lambda s: s.get("species"),
    get_bboxes=None,
    class_mapping=_CLASS_MAPPING,
)
