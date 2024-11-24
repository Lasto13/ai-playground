import { ChatOpenAI } from "@langchain/openai";

import * as tf from '@tensorflow/tfjs-core';
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { data } from "../public/data";

const llm = new ChatOpenAI({
    openAIApiKey: import.meta.env.VITE_OPENAI_KEY,
    temperature: 1,
    modelName: "gpt-4-0125-preview",
});

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
