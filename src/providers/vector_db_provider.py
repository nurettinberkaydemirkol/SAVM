import faiss
import numpy as np
import pickle
import os

class VectorDatabaseProvider:
    def __init__(self):
        self.dim = None
        self.index = None
        self.data = {}        
        self.id_order = []     

    def _rebuild_index(self):
        if self.dim is None or not self.id_order:
            return
        self.index = faiss.IndexFlatL2(self.dim)
        vectors = np.array([self.data[_id]["vector"] for _id in self.id_order], dtype='float32')
        self.index.add(vectors)

    def add_or_update(self, _id: str, vector: list, file_uri: str):
        vector_np = np.array(vector, dtype='float32')

        if self.dim is None:
            self.dim = vector_np.shape[0]
            self.index = faiss.IndexFlatL2(self.dim)

        if vector_np.shape[0] != self.dim:
            raise ValueError(f"Vektör boyutu {vector_np.shape[0]} ama beklenen {self.dim}")

        if _id not in self.data:
            self.id_order.append(_id)
        self.data[_id] = {
            "vector": vector_np,
            "file_uri": file_uri
        }
        self._rebuild_index()

    def get_by_id(self, _id: str):
        if _id not in self.data:
            return None
        entry = self.data[_id]
        return {
            "id": _id,
            "vector": entry["vector"],
            "file_uri": entry["file_uri"]
        }

    def delete(self, _id: str):
        if _id in self.data:
            del self.data[_id]
            self.id_order.remove(_id)
            self._rebuild_index()

    def search(self, query_vector: list, k=5):
        if not self.id_order:
            return []

        query = np.array(query_vector, dtype='float32').reshape(1, -1)

        if query.shape[1] != self.dim:
            raise ValueError(f"Sorgu vektörü boyutu {query.shape[1]} ama beklenen {self.dim}")

        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.id_order):
                _id = self.id_order[idx]
                entry = self.data[_id]
                results.append({
                    "id": _id,
                    "distance": float(dist),
                    "vector": entry["vector"],
                    "file_uri": entry["file_uri"]
                })
        return results

    def list_ids(self):
        return list(self.data.keys())

    def save_to_file(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if self.index:
            faiss.write_index(self.index, os.path.join(folder_path, "faiss.index"))
        meta = {
            "dim": self.dim,
            "data": self.data,
            "id_order": self.id_order
        }
        with open(os.path.join(folder_path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    def load_from_file(self, folder_path):
        index_path = os.path.join(folder_path, "faiss.index")
        meta_path = os.path.join(folder_path, "meta.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.dim = meta["dim"]
                self.data = meta["data"]
                self.id_order = meta["id_order"]