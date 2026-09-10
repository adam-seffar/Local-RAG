import chromadb
import uuid
import time

class BaseDB:
    def __init__(self, client, collection_name):
        self.client = client
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, processed_chunks):
        if not processed_chunks:
            return

        first_meta = processed_chunks[0].get("metadata", {})
        path = first_meta.get("source")

        if path and self.file_exists(path):
            print("File already exists, skipping.")
            return

        texts = [c["chunk"] for c in processed_chunks]
        embeddings = [c["embedding"] for c in processed_chunks]
        metadatas = [c["metadata"] for c in processed_chunks]
        ids = [str(uuid.uuid4()) for _ in texts]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def retrieve(self, query_embedding, n_results=5, metadata_filter=None):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter
        )

    def file_exists(self, file_path):
        results = self.collection.get(
            where={"source": file_path}
        )
        return len(results["ids"]) > 0

    def clear(self):
        self.client.delete_collection(self.collection.name)


    def update_document(self, doc_id, new_document=None, new_metadata=None):
        self.collection.update(
            ids=[doc_id],
            documents=[new_document] if new_document else None,
            metadatas=[new_metadata] if new_metadata else None
        )
        print(f"Updated document ID: {doc_id}")


    def delete_document(self, doc_id):
        self.collection.delete(ids=[doc_id])
        print(f"Deleted document ID: {doc_id}")
    

class RepositoryDB(BaseDB):
    def __init__(self, path="./repository_db"):
        client = chromadb.PersistentClient(path=path)
        super().__init__(client, "repository")
   
        
class SessionDB(BaseDB):
    def __init__(self):
        client = chromadb.Client()  
        super().__init__(client, "session_documents")


class ConversationDB:
    def __init__(self, path="./conversation_db", collection_name="conversations"):

        self.client = chromadb.PersistentClient(path = path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    
    def add_message(self, turn):
        if not message:
         return 
        message = [item["message"] for item in turn]
        speaker = [item["speaker"] for item in turn]
        ids = [str(uuid.uuid4()) for _ in range(len(message))]
        self.collection.add(ids=ids, message=message, speaker = speaker, timestamp = time.now())
        print(f"Added conversation turn to collection '{self.collection.name}'.")
    
        
    def delete_older_messages(self):
        result = self
     
   
    
    def retrieve(self, query_embedding, n_results=5, metadata_filter=None):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter
        )
        return results
    
