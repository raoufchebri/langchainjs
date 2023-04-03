import { Client } from "pg";
import { VectorStore } from "../vectorstores/base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";

interface SearchEmbeddingsResponse {
  id: number;
  content: string;
  metadata: object;
  similarity: number;
}

export interface PostgresLibArgs {
  client: Client;
  tableName?: string;
}

export class PostgresStore extends VectorStore {
  client: Client;
  tableName: string;

  constructor(embeddings: Embeddings, args: PostgresLibArgs) {
    super(embeddings, args);

    this.client = args.client;
    this.tableName = args.tableName || "documents";
  }

  async addDocuments(documents: Document[]): Promise<void> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents
    );
  }

  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    const rows = vectors.map((embedding, idx) => ({
      content: documents[idx].pageContent,
      embedding,
      metadata: documents[idx].metadata,
    }));
    try {
      const chunkSize = 200;
      for (let i = 0; i < rows.length; i += chunkSize) {
        const chunk = rows.slice(i, i + chunkSize);

        const query = `
        MERGE INTO $1 d
        USING (
          SELECT $2::text as content,
                $3::jsonb as metadata,
                $4::vector(1536) as embedding
        ) AS v(content, metadata, embedding)
        ON (d.content = v.content AND d.metadata = v.metadata)
        WHEN MATCHED THEN
          UPDATE SET embedding = v.embedding
        WHEN NOT MATCHED THEN
          INSERT (content, metadata, embedding) VALUES (v.content, v.metadata, v.embedding)
      `;

        await this.client.query(query, [this.tableName, chunk]);
      }
    } catch (error) {
      // throw new Error(
      //   `Error adding vectors: ${error.message} ${error.status} ${error.statusText}`
      // );
    }
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number
  ): Promise<[Document, number][]> {
    const qEmbeddingsStr = query.toString().replace(/\.\.\./g, "");
    // Query the database for the context
    const similaryQuery = `
      SELECT *,
      id,
      content,
      metadata,
      1 - (documents.embedding <=> '[${qEmbeddingsStr}]') as similarity
      FROM ${this.tableName}
      LIMIT ${k}
    `;
    try {
      const { rows: searches } = await this.client.query(similaryQuery, query);
      const result: [Document, number][] = (
        searches as SearchEmbeddingsResponse[]
      ).map((resp) => [
        new Document({
          metadata: resp.metadata,
          pageContent: resp.content,
        }),
        resp.similarity,
      ]);

      return result;
    } catch (error) {
      throw new Error(`Error searching for documents: ${error}`);
    }
  }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: Embeddings,
    dbConfig: PostgresLibArgs
  ): Promise<PostgresStore> {
    const docs: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });
      docs.push(newDoc);
    }
    return PostgresStore.fromDocuments(docs, embeddings, dbConfig);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: PostgresLibArgs
  ): Promise<PostgresStore> {
    const instance = new this(embeddings, dbConfig);
    await instance.addDocuments(docs);
    return instance;
  }

  static async fromExistingIndex(
    embeddings: Embeddings,
    dbConfig: PostgresLibArgs
  ): Promise<PostgresStore> {
    const instance = new this(embeddings, dbConfig);
    return instance;
  }
}
