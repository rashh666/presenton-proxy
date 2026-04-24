import { getApiUrl } from "@/utils/api";
import { ApiResponseHandler } from "./api-error-handler";
import { getHeader } from "./header";

export interface ChatMessageRequest {
  presentation_id: string;
  message: string;
}

export interface ChatMessageResponse {
  conversation_id?: string;
  response: string;
  tool_calls?: string[];
}

export class PresentationChatApi {
  static async sendMessage(
    payload: ChatMessageRequest
  ): Promise<ChatMessageResponse> {
    const response = await fetch(getApiUrl("/api/v1/ppt/chat/message"), {
      method: "POST",
      headers: getHeader(),
      body: JSON.stringify(payload),
      cache: "no-cache",
    });

    return await ApiResponseHandler.handleResponse(
      response,
      "Failed to send chat message"
    );
  }
}
