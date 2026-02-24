from collections.abc import Callable
import os
import json
import time
import tqdm
import openreview.api
import re

def scraper(prefix: str, folder: str, year_filter: Callable[[str], bool]=None):
    year_re = re.compile(r"[0-9]{4}")

    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    venues = client.get_group(id="venues").members
    venues = [i for i in venues if i.lower().startswith(prefix)]

    # venue_id = ""
    # venue_group = client.get_group(venue_id)
    # # Status keys (if accecpted, use venue_id)
    # under_review = "submission_venue_id"
    # withdrawn = "withdrawn_venue_id"
    # desk_rejected="desk_rejected_venue_id"
    # status_key_to_query = ""
    # status_id = venue_group.content[status_key_to_query]['value'] 


    for venue in tqdm.tqdm(venues, desc="Venues"):
        year = re.search(year_re, venue)
        if year is None:
            print(f"{venue}: no year")
        else:
            year = year.group()
        if year_filter is not None and not year_filter(year):
            print(f"{venue}: year {year} filtered out")
            continue
        papers = client.get_all_notes(content={"venueid": venue})
        if len(papers) == 0:
            print(f"No papers found for {venue}")

        os.makedirs(folder, exist_ok=True)
        for paper in tqdm.tqdm(papers, desc="Paper", leave=False):
            paper_data = {}
            paper_data["id"] = paper.id
            paper_data["title"] = paper.content["title"]["value"]
            paper_data["abstract"] = paper.content.get("abstract", {}).get("value", "")
            paper_data["bibtex"] = paper.content.get("_bibtex", {}).get("value", "")
            paper_data["authors"] = paper.content.get("authors", {}).get("value", [])
            paper_data["pdf"] = paper.content.get("pdf", {}).get("value", "")
            if paper_data["pdf"] != "":
                paper_data["pdf"] = "https://openreview.net" + paper_data["pdf"]
            paper_data["TLDR"] = paper.content.get("TLDR", {}).get("value", "")
            paper_data["keywords"] = paper.content.get("keywords", {}).get("value", "")
            paper_data["research_area"] = paper.content.get("research_area", {}).get("value", [])
            paper_data["year"] = year
            paper_data["venue"] = paper.content["venue"]["value"]
            paper_data["venueid"] = paper.content["venueid"]["value"]

            with open(os.path.join(folder, f"{paper.id}.json"), "w") as f:
                json.dump(paper_data, f, indent=2)
        time.sleep(2)
    
# fields
# id str
# .content["TLDR"]["value"] str
# .content["_bibtex"]["value"] str
# .content["abstract"]["value"] str
# .content["authors"]["value"] list[str]
# .content["keywords"]["value"] str
# .content["pdf"]["value"] https://openreview.net + str
# .content["research_area"]["value"] list[str]
# .content["title"]["value"] str
# year from venue
# {'cdate': 1711193824635,
#  'content': {'TLDR': {'value': 'The Khayyam Challenge offers an evaluation '
#                                'framework for Persian-supporting LLMs, '
#                                'featuring 20,805 original, diverse questions '
#                                'accompanied by rich metadata.'},
#              '_bibtex': {'value': '@inproceedings{\n'
#                                   'ghahroodi2024khayyam,\n'
#                                   'title={Khayyam Challenge (Persian{MMLU}): '
#                                   'Is Your {LLM} Truly Wise to The Persian '
#                                   'Language?},\n'
#                                   'author={Omid Ghahroodi and Marzia Nouri and '
#                                   'Mohammad Vali Sanian and Alireza Sahebi and '
#                                   'Doratossadat Dastgheib and Ehsaneddin '
#                                   'Asgari and Mahdieh Soleymani Baghshah and '
#                                   'Mohammad Hossein Rohban},\n'
#                                   'booktitle={First Conference on Language '
#                                   'Modeling},\n'
#                                   'year={2024},\n'
#                                   'url={https://openreview.net/forum?id=yIEyHP7AvH}\n'
#                                   '}'},
#              'abstract': {'value': 'Evaluating Large Language Models (LLMs) is '
#                                    'challenging due to their generative '
#                                    'nature, necessitating precise evaluation '
#                                    'methodologies. Additionally, non-English '
#                                    'LLM evaluation lags behind English, '
#                                    'resulting in the absence or weakness of '
#                                    'LLMs for many languages.\n'
#                                    'In response to this necessity, we '
#                                    'introduce Khayyam Challenge (also known as '
#                                    'PersianMMLU), a meticulously curated '
#                                    'collection comprising 20,805 four-choice '
#                                    'questions sourced from 38 diverse tasks '
#                                    'extracted from Persian examinations, '
#                                    'spanning a wide spectrum of subjects, '
#                                    'complexities, and ages. The primary '
#                                    'objective of the Khayyam Challenge is to '
#                                    'facilitate the rigorous evaluation of LLMs '
#                                    'that support the Persian language. '
#                                    'Distinctive features of the Khayyam '
#                                    'Challenge are (i) its comprehensive '
#                                    'coverage of various topics, including '
#                                    'literary comprehension, mathematics, '
#                                    'sciences, logic, intelligence testing, etc '
#                                    'aimed at assessing different facets of '
#                                    'LLMs such as language comprehension, '
#                                    'reasoning, and information retrieval '
#                                    'across various educational stages, from '
#                                    'lower primary school to upper secondary '
#                                    'school (ii) its inclusion of rich metadata '
#                                    'such as human response rates, difficulty '
#                                    'levels, and descriptive answers (iii) its '
#                                    'utilization of new data to avoid data '
#                                    'contamination issues prevalent in existing '
#                                    'frameworks (iv) its use of original, '
#                                    'non-translated data tailored for Persian '
#                                    'speakers, ensuring the framework is free '
#                                    'from translation challenges and errors '
#                                    'while encompassing cultural nuances (v) '
#                                    'its inherent scalability for future data '
#                                    'updates and evaluations without requiring '
#                                    'special human effort. Previous works '
#                                    'lacked an evaluation framework that '
#                                    'combined all of these features into a '
#                                    'single comprehensive benchmark. '
#                                    'Furthermore, we evaluate a wide range of '
#                                    'existing LLMs that support the Persian '
#                                    'language, with statistical analyses and '
#                                    'interpretations of their outputs. We '
#                                    'believe that the Khayyam Challenge will '
#                                    'improve advancements in LLMs for the '
#                                    'Persian language by highlighting the '
#                                    'existing limitations of current models, '
#                                    'while also enhancing the precision and '
#                                    'depth of evaluations on LLMs, even within '
#                                    'the English language context.'},
#              'author_guide': {'value': ['I certify that this submission '
#                                         'complies with the submission '
#                                         'instructions as described on '
#                                         'https://colmweb.org/AuthorGuide.html']},
#              'authorids': {'value': ['~Omid_Ghahroodi1',
#                                      '~Marzia_Nouri1',
#                                      '~Mohammad_Vali_Sanian1',
#                                      '~Alireza_Sahebi1',
#                                      '~Doratossadat_Dastgheib1',
#                                      '~Ehsaneddin_Asgari1',
#                                      '~Mahdieh_Soleymani_Baghshah1',
#                                      '~Mohammad_Hossein_Rohban1']},
#              'authors': {'value': ['Omid Ghahroodi',
#                                    'Marzia Nouri',
#                                    'Mohammad Vali Sanian',
#                                    'Alireza Sahebi',
#                                    'Doratossadat Dastgheib',
#                                    'Ehsaneddin Asgari',
#                                    'Mahdieh Soleymani Baghshah',
#                                    'Mohammad Hossein Rohban']},
#              'code_of_ethics': {'value': ['I acknowledge that I and all '
#                                           'co-authors of this work have read '
#                                           'and commit to adhering to the COLM '
#                                           'Code of Ethics on '
#                                           'https://colmweb.org/CoE.html']},
#              'keywords': {'value': 'LLM, Evaluation, Multitask, Persian '
#                                    'Language Understanding'},
#              'paperhash': {'value': 'ghahroodi|khayyam_challenge_persianmmlu_is_your_llm_truly_wise_to_the_persian_language'},
#              'pdf': {'value': '/pdf/547ec1407b182679aaa3d8c4773815a3d5ba8040.pdf'},
#              'research_area': {'value': ['Data', 'Evaluation']},
#              'title': {'value': 'Khayyam Challenge (PersianMMLU): Is Your LLM '
#                                 'Truly Wise to The Persian Language?'},
#              'venue': {'value': 'COLM'},
#              'venueid': {'value': 'colmweb.org/COLM/2024/Conference'}},
#  'ddate': None,
#  'details': {'writable': False},
#  'domain': 'colmweb.org/COLM/2024/Conference',
#  'forum': 'yIEyHP7AvH',
#  'id': 'yIEyHP7AvH',
#  'invitations': ['colmweb.org/COLM/2024/Conference/-/Submission',
#                  'colmweb.org/COLM/2024/Conference/-/Post_Submission',
#                  'colmweb.org/COLM/2024/Conference/Submission1499/-/Revision',
#                  'colmweb.org/COLM/2024/Conference/-/Edit',
#                  'colmweb.org/COLM/2024/Conference/Submission1499/-/Camera_Ready'],
#  'license': 'CC BY 4.0',
#  'mdate': 1724633541476,
#  'nonreaders': None,
#  'number': 1499,
#  'odate': 1724633541461,
#  'parent_invitations': None,
#  'pdate': 1720608930830,
#  'readers': ['everyone'],
#  'replyto': None,
#  'signatures': ['colmweb.org/COLM/2024/Conference/Submission1499/Authors'],
#  'tcdate': 1711193824635,
#  'tmdate': 1724633541476,
#  'writers': ['colmweb.org/COLM/2024/Conference',
#              'colmweb.org/COLM/2024/Conference/Submission1499/Authors']}