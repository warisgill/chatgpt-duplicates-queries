import json
from datetime import datetime
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
from diskcache import Index
import fire 


class ChatGPTInteractionData:
    def __init__(self, fpath):
        self.data = self.__readJsonFile(fpath)
        self.rows = []
        self.df = self.__processedDF()

    def getDF(self):
        return self.df

    def __readJsonFile(self, file_path):
        conversations = None
        with open(file_path, "r") as file:
            conversations = json.load(file)
        return conversations

    def __processedDF(self):
        def _lambda(conversation):
            for u_k, message_details in conversation["mapping"].items():
                if message_details["message"] is not None:  # Check if there's a message
                    author_role = message_details["message"]["author"]["role"]
                    if author_role == "user":
                        user_query_dict, response_key = self.__userQuery(
                            message_details
                        )
                        if user_query_dict is None:
                            continue
                        res_dict = {}
                        if response_key is not None:
                            if 'parts' not in conversation["mapping"][response_key]["message"]["content"]:
                                continue

                            res_dict = self.__getResponseDetails(
                                conversation["mapping"][response_key]
                            )
                        user_query_dict.update(res_dict)
                        if "res_time" in user_query_dict:
                            user_query_dict["Response Time"] = (
                                user_query_dict["res_time"]
                                - user_query_dict["query_time"]
                            ).total_seconds()
                        else:
                            user_query_dict["Response Time"] = None
                        all_dict.append(user_query_dict)

        all_dict = []
        _ = [_lambda(conv) for conv in tqdm(self.data)]

        return pd.DataFrame(all_dict)

    def __userQuery(self, message_details):
        d = {}
        content = None
        try:
            content = message_details["message"]["content"]["parts"][0].strip()
        except:
            return {}, None
        
        
        create_time = message_details["message"]["create_time"]
        d["user_query"] = content
        # d['correct_user_query'] = TextBlob(content).correct()
        d["query_time"] = datetime.fromtimestamp(create_time)
        d["query_size"] = len(content.split())
        d["query_type"] = message_details["message"]["content"]["content_type"]
        d["query_status"] = message_details["message"]["status"]
        children = message_details["children"]
        d["query_total_responses"] = len(children)

        if len(children) == 0:
            return d, None        
        assert len(children) > 0 and len(children) < 10
        response_key = children[0]
        return d, response_key

    def __getResponseDetails(self, response_details):
        d = {}
        # print(f'Res {response_details["message"]["content"]}')
        assert len(response_details["message"]["content"]["parts"]) == 1
        content = response_details["message"]["content"]["parts"][0].strip()
        create_time = response_details["message"]["create_time"]

        d["res_text"] = content
        d["res_time"] = datetime.fromtimestamp(create_time)
        d["res_size"] = len(content.split())
        d["res_type"] = response_details["message"]["content"]["content_type"]
        d["res_status"] = response_details["message"]["status"]
        d["res_author_role"] = response_details["message"]["author"]["role"]
        d["res_weight"] = response_details["message"]["weight"]
        d["res_end_turn"] = response_details["message"]["end_turn"]

        if "finish_details" in response_details["message"]["metadata"]:
            d["res_finish_details"] = response_details["message"]["metadata"][
                "finish_details"
            ]["type"]
            if d["res_finish_details"].find("stop") != -1:
                d["res_is_complete"] = True
            else:
                d["res_is_complete"] = False
        return d


class FindDuplicateQueries:
    def __init__(self, df, query_size, cos_sim_threshold):
        self.cache = Index(".embs_cache")
        self.df = df
        # select queries with size less than 256
        print(f"> Total queries: {len(self.df)}")
        self.df = self.df[self.df["query_size"] < query_size]
        print(f"> Total queries with size less than {query_size}: {len(self.df)}")

        self.mname = "nomic-ai/nomic-embed-text-v1.5"
        self.cos_sim_threshold = cos_sim_threshold
        self.__filterQueries()
        self.embeddings = self.__getEmbeddings(self.df["user_query"].values)

    def __nomicEmbedding(self, qs):
        matryoshka_dim = 768
        new_qs = []
    
        print(f"> checking embeddings from local storage")
        new_qs = [ q for q in tqdm(qs) if q not in list(self.cache.keys())]

        if len(new_qs) > 0:
            model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
            )
            print(">> Embedding new queries")
            # for q in tqdm(new_qs):
            #     print(f"Embedding {q}")
            embeddings = model.encode(
                new_qs,
                convert_to_tensor=True,
                show_progress_bar=True,
                device="cpu",
                batch_size=12,
                
            )
            embeddings = F.layer_norm(
                embeddings, normalized_shape=(embeddings.shape[1],)
            )
            embeddings = embeddings[:, :matryoshka_dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)

            print(">> storing new embeddings in cache") 
            for i, k in tqdm(enumerate(new_qs)):
                self.cache[k] = embeddings[i]
        else:
            print(">> Using embeddings from cache")

        embeddings = [self.cache[k] for k in tqdm(qs)]
        return embeddings

    def __getEmbeddings(self, qs):
        if self.mname.find("nomic-ai/nomic-embed-text-v1.5") != -1:
            return self.__nomicEmbedding(qs)
        else:
            raise Exception("Model not found")

    def __filterQueries(self):
        ingore_queries = []
        with open("ignore_prompts.txt", "r") as file:
            ingore_queries = file.readlines()
            ingore_queries = [q.strip() for q in ingore_queries]

        embeds_ignore = self.__getEmbeddings(ingore_queries)
        embeds = self.__getEmbeddings(self.df["user_query"].values)

        ignore_indices = []
        for i, qemb in enumerate(embeds):
            for j, q_ignore_em in enumerate(embeds_ignore):
                sim = util.pytorch_cos_sim(qemb, q_ignore_em)
                if sim > 0.70:
                    ignore_indices.append(i)
                    break

        print(f"Total queries {len(self.df)}")
        self.df = self.df[~self.df.index.isin(ignore_indices)]
        print(f"Total queries after filtering from ingore list {len(self.df)}")


    def getDuplicateQueries(self):
        def _lambda(qid):
            row_dict = {}
            row_dict["corpus_id"] = qid
            row_dict["query"] = self.df.iloc[qid]["user_query"]
            temp_li = [
                d | {"query": self.df.iloc[d["corpus_id"]]["user_query"]}
                for d in li_of_li_of_dict[qid]
                if d["corpus_id"] != qid and d["score"] > self.cos_sim_threshold
            ]
            row_dict["duplicates"] = temp_li
            dup_set.update([d["corpus_id"] for d in temp_li])
            return row_dict

        dup_set = set()
        li_of_li_of_dict = util.semantic_search(
            self.embeddings, self.embeddings, top_k=len(self.embeddings)
        )
        assert len(li_of_li_of_dict) == len(self.embeddings) and len(
            li_of_li_of_dict[0]
        ) == len(self.embeddings), "Invalid length"
        user_inspection_pairs = [_lambda(qid) for qid in tqdm(range(len(self.embeddings))) ]
        
        # to get the duplicates and skipping the ones already found
        dup_set = set()
        _ = [_lambda(qid) for qid in tqdm(range(len(self.embeddings))) if qid not in dup_set]

        return pd.DataFrame(user_inspection_pairs), len(dup_set)


def main(qsize=256, cos_sim=0.8):
    file_path = "private_csv_json/conversations.json"
    output_fpath = "private_csv_json/all_queries.csv"
    final_results_fpath = "result_csv/final_results.csv"

    result_dict = {}

    gpt_data = ChatGPTInteractionData(file_path)
    df = gpt_data.getDF()
    df.to_csv(output_fpath, index=False)
    
    df = pd.read_csv(output_fpath)
    df.dropna(subset=["user_query"], inplace=True)

    df["query_time"] = pd.to_datetime(df["query_time"])

    min_date  =df["query_time"].min()
    max_date = df["query_time"].max()

    print(f"Min date: {min_date}")
    print(f"Max date: {max_date}")

    number_of_days = (max_date - min_date).days

    dup = FindDuplicateQueries(df, query_size=qsize, cos_sim_threshold=cos_sim)
    dup_df, total_duplicates = dup.getDuplicateQueries()
    
    result_dict["total_queries_without_filter"] = len(df)
    result_dict["total_filterd_quieres"] = len(dup_df)
    result_dict["total_duplicates"] = total_duplicates
    result_dict["total_duplicates_percentage"] = total_duplicates/len(dup_df)
    result_dict['query_size'] = qsize
    result_dict['cos_sim_threshold'] = cos_sim
    result_dict['min_date'] = min_date
    result_dict['max_date'] = max_date
    result_dict['number_of_days'] = number_of_days

    # print(f"Total duplicates found: {total_duplicates/len(dup_df)}")

    result_df = pd.DataFrame([result_dict])

    result_df.to_csv(final_results_fpath, index=False) 


    print(result_df)

    only_dup_df = dup_df[dup_df["duplicates"].apply(len) > 0]
    dup_df.to_csv("private_csv_json/scores_potent_duplicates.csv", index=False)
    only_dup_df.to_csv("private_csv_json/only_duplicates.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)

