import pyglottolog
import os
import dotenv
import pyglottolog.config
import pyglottolog.languoids
dotenv.load_dotenv()
import functools
import yaml


@functools.lru_cache(maxsize=1)
def get_glottolog_instance():
    glottolog = pyglottolog.Glottolog(os.getenv('GLOTTOLOG_REPOS'), cache=True)
    return glottolog

@functools.lru_cache(maxsize=1)
def get_language_index():
    glottolog = get_glottolog_instance()
    lang_index = {l.name: l for l in glottolog.languoids()}
    return lang_index

@functools.lru_cache(maxsize=1)
def get_lang_map(path: str = 'lang_map.yml'):
    with open(path, 'r', encoding='utf-8') as f:
        lang_map = yaml.safe_load(f)
    return lang_map
    

CHINA = pyglottolog.languoids.Country(id='CN', name='China')
INDIA = pyglottolog.languoids.Country(id='IN', name='India')
EURASIA = get_glottolog_instance().macroareas['eurasia']
AFRICA = get_glottolog_instance().macroareas['africa']
NO_MACROAREA = pyglottolog.config.Macroarea(id='None', name='None', description='No macroarea assigned', reference_id='')

@functools.lru_cache(maxsize=1)
def get_european_countries():
    """Return a list of European country ISO alpha-2 codes."""
    ids = ["AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI", "LT", "LU", "MT", "MD", "MC", "ME", "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA", "GB", "VA", "CY"]    
    # Cyprus included: (Cypriot) Greek is European, Turkish is spoken in other European countries
    # Kosovo missing from Glottolog (Serbian-Croatian-Bosnian)
    # Russia: languages spoken in Russia and other countries are in the continent of the other countries (e.g., Polish, Georgian)
        # Languages only spoken in Russia:
    glottolog = get_glottolog_instance()
    europe = tuple([i for i in glottolog.countries if i.id in ids])
    return europe

class LanguoidPlaceHolder:
    def __init__(self, name: str, macroareas: list, countries: list, family = None):
        self.name = name
        self.macroareas = macroareas
        self.countries = countries
        self.category: str | None = None
        self.family = family # .name
        self.endangerment = None # .status.name

CHINESE = LanguoidPlaceHolder("Chinese", [EURASIA], [CHINA], family=get_glottolog_instance().languoid("sino1245")) # Sino-Tibetan
AFRICAN = LanguoidPlaceHolder("African Languages", [AFRICA], [])
INDIAN = LanguoidPlaceHolder("Indian Languages", [EURASIA], [INDIA])
NO_LANGUAGE = LanguoidPlaceHolder("None", [NO_MACROAREA], [])

# Anything related to Chinese languages not in glottolog
CHINESE_SET = {    
    'Ancient Chinese',
    'Chinese',
    'Chinese-simplified',
    'Chinese-traditional',
    'Late Middle Chinese',
    'Modern Chinese',
    'Standard Written Chinese',
    'Traditional Chinese',
    'Han-Vi'
}

AFRICAN_SET = {
    'African Languages',
}

INDIAN_SET = {
    'Indian Languages',
}

def map_to_glottolog(lang_name: str) -> pyglottolog.languoids.Languoid | LanguoidPlaceHolder | None:
    index_ = get_language_index()
    lang_map = get_lang_map()
    if lang_name in CHINESE_SET:
        return CHINESE
    elif lang_name in AFRICAN_SET:
        return AFRICAN
    elif lang_name in INDIAN_SET:
        return INDIAN
    languoid = None

    languoid = index_.get(lang_name)
    if languoid is None:
        mapped_name = lang_map.get(lang_name) 
        if mapped_name is not None:
            languoid = index_.get(mapped_name)
    if languoid is None:
        languoid = NO_LANGUAGE
    return languoid        

def get_macroarea(lang: pyglottolog.languoids.Languoid | LanguoidPlaceHolder | None, replace_map: dict[str, str] = {}) -> list[str]:
    if lang is None:
        return [replace_map.get(NO_MACROAREA.name, NO_MACROAREA.name)]
    else:
        names = []
        for macroarea in lang.macroareas:              
            names.append(replace_map.get(macroarea.name, macroarea.name))
    return names

def is_european(lang: pyglottolog.languoids.Languoid | LanguoidPlaceHolder | None) -> bool:
    if lang is None:
        return False
    european_countries = get_european_countries()
    return any(country in european_countries for country in lang.countries) and len(lang.countries) > 0

def is_exclusive_russian(lang: pyglottolog.languoids.Languoid | LanguoidPlaceHolder | None) -> bool:
    if lang is None:
        return False
    return all(country.id == 'RU' for country in lang.countries) and len(lang.countries) > 0

def get_family(lang: pyglottolog.languoids.Languoid | LanguoidPlaceHolder | None) -> str | None:    
    if lang is None or lang.family is None:
        return None
    else:
        return lang.family.name
        
def get_endangerment(lang: pyglottolog.languoids.Languoid | LanguoidPlaceHolder | None) -> str | None:
    if lang is None or lang.endangerment is None:
        return None
    elif lang.endangerment.status is None:
        return None
    else:
        return lang.endangerment.status.name