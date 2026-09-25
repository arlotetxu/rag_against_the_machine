    def bm25_index(self) -> BM25Okapi:
        """
        0.9, 1.2, 1.5, 1.8,
        , 0.5, 0.75, 0.9
        """
        corpus_tokens = self.tokenize_chunks()
        # for k1 in (1.9, 2.0, 2.1, 2.2):
        #     bm25_index = BM25Okapi(
        #         corpus_tokens, k1=k1, b=0.3)  # type: ignore[no-untyped-call]
        #     self.save_index_chunks(bm25_index)
        #     ic(k1)
        #     Retrieval().get_batch_query_chunks(
        #         "data/datasets/private/AnsweredQuestions/dataset_docs_private.json",
        #         10,
        #         "data/output/AnsweredQuestions/dataset_docs_private.json")
        #     Evaluate().get_recall(
        #         "data/output/AnsweredQuestions/dataset_docs_private.json",
        #         "data/datasets/private/AnsweredQuestions/dataset_docs_private.json",
        #         10)
        bm25_index = BM25Okapi(
            corpus_tokens, k1=2.0, b=0.3)  # type: ignore[no-untyped-call]
        return bm25_index
