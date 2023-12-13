from mteb import MTEB
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from sentence_transformers import SentenceTransformer

class SpanishRedditClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "SpanishRedditClustering",
            "hf_hub_name": "willhath/spanish-reddit-clustering", 
            "description": "Clustering of Spanish Reddit posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["es"], 
            "main_score": "v_measure",
        }

class SpanishTwentyNewsgroupsClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "SpanishTwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/spanish-twentynewsgroups-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of Spanish 20 Newsgroups posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["es"], 
            "main_score": "v_measure",
        }

class FrenchRedditClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FrenchRedditClustering",
            "hf_hub_name": "willhath/french-reddit-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of French Reddit posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["fr"], 
            "main_score": "v_measure",
        }


    
class FrenchTwentyNewsgroupsClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FrenchTwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/french-twentynewsgroups-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of French 20 Newsgroups posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["fr"], 
            "main_score": "v_measure",
        }
    

class SwahiliRedditClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "SwahiliRedditClustering",
            "hf_hub_name": "willhath/swahili-reddit-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of Swahili Reddit posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["sw"], 
            "main_score": "v_measure",
        }
    
class SwahiliTwentyNewsgroupsClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "SwahiliTwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/swahili-twentynewsgroups-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of Swahili 20 Newsgroups posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["sw"], 
            "main_score": "v_measure",
        }

    
class GermanRedditClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "GermanRedditClustering",
            "hf_hub_name": "willhath/german-reddit-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of German Reddit posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["de"], 
            "main_score": "v_measure",
        }
    
    
class GermanTwentyNewsgroupsClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "GermanTwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/german-twentynewsgroups-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of German 20 Newsgroups posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"],
            "eval_langs": ["de"], 
            "main_score": "v_measure",
        }
    
class TurkishRedditClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TurkishRedditClustering",
            "hf_hub_name": "willhath/turkish-reddit-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of Turkish Reddit posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"], 
            "eval_langs": ["tr"],
            "main_score": "v_measure",
        }
    
class TurkishTwentyNewsgroupsClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TurkishTwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/turkish-twentynewsgroups-clustering",  # Replace with your actual Hugging Face Hub name
            "description": "Clustering of Turkish 20 Newsgroups posts",
            "reference": "None",  # Replace with the actual reference
            "type": "Clustering",
            "category": "text",
            "eval_splits": ["test"], 
            "eval_langs": ["tr"],
            "main_score": "v_measure",
        }
