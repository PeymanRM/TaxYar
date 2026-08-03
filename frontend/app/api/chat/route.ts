import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import {
  streamText,
  convertToModelMessages,
  type UIMessage,
  type JSONSchema7,
} from "ai";

const myProvider = createOpenAICompatible({
  name: "TaxYar",
  baseURL: "http://localhost:8000",
  apiKey: "",
});

export async function POST(req: Request) {
  const {
    messages,
    system,
    tools,
    model = "taxyar",
  }: {
    messages: UIMessage[];
    system?: string;
    model?: string;
    tools?: Record<string, { description?: string; parameters: JSONSchema7 }>;
  } = await req.json();

  const result = streamText({
    model: myProvider.chatModel(model),
    messages: await convertToModelMessages(messages),
    system,
    tools: {
      ...frontendTools(tools ?? {}),
    },
  });

  return result.toUIMessageStreamResponse();
}