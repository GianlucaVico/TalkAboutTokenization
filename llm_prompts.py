SYSTEM_PROMPT_GENERIC = "You are a helpful assistant."

SYSTEM_PROMPT_FILTER = ( 
    "You are a helpful assistant. Your task is to decide whether the following paper is about text tokenization."
    "This includes tokenization, subword segmentation, tokenizer free methods, pixel language models, etc."
    "You are given the title and abstract of the paper."    
)

SYSTEM_PROMPT_UNIT = (
    "You are a helpful assistant. Your task is to identify the basic unit of analysis a text tokenizer uses, "
    "based on the paper’s title and content. The possible options are: bytes, characters, subwords, "
    "words, morphemes, patches, pixels, or other (specify)."
    "Answer only with a list of the most appropriate units, each on its own line formatted as '* {unit}'."
)

SYSTEM_PROMPT_LANGUAGE_MULTILINGUAL = (
    "You are a helpful assistant. Your task is to identify the languages that are used in the experiments of a research paper on text tokenization, "
    "based on the paper’s title and content."     
    "Answer only with the list of languages, each on its own line formatted as '* {language}'."
    "If not language is specified, answer with 'Not specified'. "
    "If the languages is not explicitly specified but can be inferred from the content, add '(inferred)' after the language name."
)

SYSTEM_PROMPT_LANGUAGE_SPECIFIC = ( # Is this paper about a specific language? If so, which one?
    "You are a helpful assistant. Your task is to decide whether the text tokenization paper focuses on one or more specific languages, "
    "based on the paper’s title and content. If it does, provide the name of the language; if not, respond with 'No'."
    "For example, if the paper is about tokenization in Chinese, Japanese, and Korean, the answer should be 'Chinese, Japanese, Korean'. " 
    "But if the paper is about tokenization in general and it is tested on multiple languages, the answer should be 'No'."
    "Answer only with the language name (or 'No')."
)

SYSTEM_PROMPT_EVALUATION_INTRINSIC = ( # Is the tokenizer evaluated intrinsically? If so, what are the evaluation metrics?
    "You are a helpful assistant. Your task is to determine if the text tokenizer described in the paper is evaluated intrinsically, "
    "and if so, identify the evaluation metrics used, based on the paper’s title and content."
    "Answer only with a list of the evaluation metrics, each on its own line formatted as '* {metric}'. "
    "If the paper does not perform intrinsic evaluation, answer with 'No'. "
    "Examples of intrinsic metrics include: fertility, morphological accuracy, tokenization parity, etc."
)

SYSTEM_PROMPT_EVALUATION_EXTRINSIC = ( # Is the tokenizer evaluated extrinsically? If so, what is the downstream task?
    "You are a helpful assistant. Your task is to determine if the text tokenizer described in the paper is evaluated extrinsically, "
    "and if so, identify the downstream tasks used for evaluation, based on the paper’s title and content."
    "Answer only with a list of the downstream tasks, each on its own line formatted as '* {task}'. "
    "If the paper does not perform extrinsic evaluation, answer with 'No'. "
    "Examples of downstream tasks include: language modeling, machine translation, part-of-speech tagging, etc."
)

SYSTEM_PROMPT_MOTIVATION = (
    "You are a helpful assistant. Your task is to identify the motivations behind a research paper on text tokenization, based on the paper’s title and content."
)

####################################################################

USER_PROMPT_TITLE_ABSTRACT_YN = (
    "The title is: '{title}'.\n"
    "The abstract is: '{abstract}'.\n"
    "Answer only with 'Yes' or 'No'."
)

USER_PROMPT_UNIT = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"
    "What is the basic unit of analysis used by the text tokenizer described in the paper? "
    "Answer only with a list of the most appropriate units, each on its own line formatted as '* {unit}'."
)

USER_PROMPT_LANGUAGE_MULTILINGUAL = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"
    "What are the languages that are used in the experiments of this research paper on text tokenization? "
    "Answer only with the list of languages, each on its own line formatted as '* {language}'."    
)

USER_PROMPT_LANGUAGE_LANGUAGE_SPECIFIC = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"
    "Does the paper focus on one or more specific languages? "
    "If it does, provide the name of the languages; if not, respond with 'No'. "
)

USER_PROMPT_EVALUATION_INTRINSIC = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"
    "Is the text tokenizer described in the paper evaluated intrinsically? If so, what are the evaluation metrics used? "
    "Answer only with a list of the evaluation metrics, each on its own line formatted as '* {metric}'. "
    "If the paper does not perform intrinsic evaluation, answer with 'No'. "
)

USER_PROMPT_EVALUATION_EXTRINSIC = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"
    "Is the text tokenizer described in the paper evaluated extrinsically? If so, what are the downstream tasks used for evaluation? "
    "Answer only with a list of the downstream tasks, each on its own line formatted as '* {task}'. "
    "If the paper does not perform extrinsic evaluation, answer with 'No'. "
)

USER_PROMPT_MOTIVATION = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"    
    "Briefly summarize in a single sentence the main motivations behind the research presented in the paper." 
    "The main motivation behind the research presented in the paper is..."
)

TOKENIZATION_EXAMPLES = (
    "These are two examples with title and abstract:\n\n"
    "The title is: 'ADASPLASH: Adaptive Sparse Flash Attention'.\n"
    "The abstract is: 'The computational cost of softmax-based attention in transformers limits their applicability to long-context tasks. Adaptive sparsity, of which α-entmax attention is an example, offers a flexible data-dependent alternative, but existing implementations are inefficient and do not leverage the sparsity to obtain runtime and memory gains. In this work, we propose ADASPLASH, which combines the efficiency of GPU-optimized algorithms with the sparsity benefits of α-entmax. We first introduce a hybrid Halley-bisection algorithm, resulting in a 7-fold reduction in the number of iterations needed to compute the α-entmax transformation. Then, we implement custom Triton kernels to efficiently handle adaptive sparsity. Experiments with RoBERTa and ModernBERT for text classification and single-vector retrieval, along with GPT-2 for language modeling, show that our method achieves substantial improvements in runtime and memory efficiency compared to existing α-entmax implementations. It approachesand in some cases surpasses—the efficiency of highly optimized softmax implementations like FlashAttention-2, enabling long-context training while maintaining strong task performance.'.\n"
    "The answer is: No.\n\n"
    "The title is: 'Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates'.\n"
    "The abstract is: 'Subword units are an effective way to alleviate the open vocabulary problems in neural machine translation (NMT). While sentences are usually converted into unique subword sequences, subword segmentation is potentially ambiguous and multiple segmentations are possible even with the same vocabulary. The question addressed in this paper is whether it is possible to harness the segmentation ambiguity as a noise to improve the robustness of NMT. We present a simple regularization method, subword regularization, which trains the model with multiple subword segmentations probabilistically sampled during training. In addition, for better subword sampling, we propose a new subword segmentation algorithm based on a unigram language model. We experiment with multiple corpora and report consistent improvements especially on low resource and out-of-domain settings.'.\n"
    "The answer is: Yes.\n\n"    
)
