import { ChatOpenAI } from "@langchain/openai";

import * as tf from '@tensorflow/tfjs-core';
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {ChatPromptTemplate, PromptTemplate} from "@langchain/core/prompts";
import { data } from "../public/data";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {StringOutputParser} from "@langchain/core/output_parsers";

const llm = new ChatOpenAI({
    openAIApiKey: import.meta.env.VITE_OPENAI_KEY,
    temperature: 1,
    modelName: "gpt-4-0125-preview",
});

const vectorStore = await generateAndStoreEmbeddings();

export async function generateAndStoreEmbeddings() {
    await tf.ready();

    const file = data

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 100,
    });

    const docs = await textSplitter.createDocuments([file]);

    const vectorStore = await MemoryVectorStore.fromDocuments(
        docs,
        new TensorFlowEmbeddings()
    );

    return vectorStore
}

export const generateAnswerEmbeddings = async (question: string) => {
    const prompt = PromptTemplate.fromTemplate(` Use the following pieces of context to answer the question at the end.
If you can't find the answer in the provided context, just say that you cannot answer the question based on the provided context, 
don't answer based on your training data or hallucinate.

{context}

Question: {question}

Helpful Answer:`);

    let answer = ''
    try {
        const customRagChain = await createStuffDocumentsChain({
            llm,
            prompt,
            outputParser: new StringOutputParser(),
        });

        const retriever = vectorStore.asRetriever();
        const context = await retriever.invoke(question);

        answer = await customRagChain.invoke({
            question,
            context,
        });

    } catch (e) {
        console.log({ e })
        return 'Something went wrong'
    }
    return answer
}

export async function generateAnswerFSP(
    question: string,
    promptTemplate: string = "You are a travel assistant.",
) {
    // Keep examples simple - the model should learn from their format
    const examples = [
        {
            input: 'What are the best restaurants in Amsterdam?',
            output: 'The highest rated restaurants in Amsterdam are (1) (2) (3)'
        },
        {
            input: 'What are the must-visit museums in Amsterdam?',
            output: 'The top museums in Amsterdam are (1) Rijksmuseum (2) Van Gogh Museum (3) Anne Frank House'
        }
    ];

    const messages = [
        {
            role: "system",
            content: promptTemplate
        },
        ...examples.flatMap(example => [
            { role: "user", content: example.input },
            { role: "assistant", content: example.output }
        ]),
        { role: "user", content: question }
    ];

    try {
        const result = await llm.invoke(messages);
        return result.content as string;
    } catch (e) {
        console.log('Error:', e);
        return 'Something went wrong';
    }
}

export async function generateAnswer(
    question: string,
    promptTemplate: string = "Take the role of a {role}, that answers questions in a {style} way.",
    role: string = "Personal travel assistant",
    style: string = "consistent"
) {
    let answer = ''

    const chatPrompt = ChatPromptTemplate.fromMessages([
        ["system", promptTemplate],
        ["human", "{question}"],
    ])

    const formattedPrompt = await chatPrompt.formatMessages({
        role,
        style,
        question
    });
    
    try {
        const result = await llm.invoke(formattedPrompt);

        answer = result?.content as string

    } catch (e) {
        console.log(e);
        return 'Something went wrong'
    }

    return answer
}
