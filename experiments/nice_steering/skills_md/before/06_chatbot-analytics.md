# 06. chatbot-analytics (before)

- skill_id: `a5e75fa1-be3d-42fc-b278-a37de3c7c26e`
- target_token: `NICE`
- token_propagation_before: `0.0000`

## Before Skill

- skill_id: `a5e75fa1-be3d-42fc-b278-a37de3c7c26e`
- name: `chatbot-analytics`
- description: Implement AI chatbot analytics and conversation monitoring. Use when adding conversation metrics, tracking AI usage, measuring user engagement with chat, or building conversation dashboards. Activates for AI analytics, token tracking, conversation categorization, and chat performance.
- tags: []

### capabilities

- `cap_1`: Implement AI chatbot analytics and conversation monitoring. Use when adding conversation metrics, tracking AI usage, measuring user engagement with chat, or building conversation dashboards. Activates for AI analytics, token tracking, conversation categorization, and chat performance.

### parameters

- (none)

### examples

- (none)

### body

---
name: chatbot-analytics
description: Implement AI chatbot analytics and conversation monitoring. Use when adding conversation metrics, tracking AI usage, measuring user engagement with chat, or building conversation dashboards.
  Activates for AI analytics, token tracking, conversation categorization, and chat performance.
allowed-tools: Read,Write,Edit,Bash(npm:*,npx:*)
metadata:
  category: Data & Analytics
  tags:
  - analytics
  - chatbot
  - ai-metrics
  pairs-with:
  - skill: llm-streaming-response-handler
    reason: Streaming response metrics (TTFT, tokens/sec) are key chatbot analytics dimensions
  - skill: data-pipeline-engineer
    reason: Conversation data pipelines feed analytics dashboards and reporting systems
  - skill: prompt-engineer
    reason: Analytics on prompt effectiveness drive iterative prompt optimization
---

# AI Chatbot Analytics

This skill helps you implement analytics for the AI coaching chat feature while maintaining HIPAA compliance.

## Core Metrics to Track

Based on [industry best practices](https://hiverhq.com/blog/chatbot-analytics), track these 13 key metrics:

| Metric | Description | HIPAA Safe? |
|--------|-------------|-------------|
| Total Sessions | Number of chat sessions | Yes |
| Avg Messages/Session | Messages per conversation | Yes |
| Avg Session Duration | Time spent in chat | Yes |
| Engagement Rate | % users who use chat | Yes |
| Completion Rate | Sessions ended naturally | Yes |
| Abandonment Rate | Sessions ended early | Yes |
| Response Time | AI response latency | Yes |
| Token Usage | Total/avg tokens consumed | Yes |
| Error Rate | Failed responses | Yes |
| Fallback Rate | "I don't understand" responses | Yes |
| Topic Categories | What users discuss | Metadata only |
| Sentiment Trend | Emotional direction | Derived only |
| Crisis Triggers | Emergency detection | Metadata only |

## HIPAA-Compliant Analytics

### What to Track

```typescript
// Conversation metadata (SAFE)
interface ConversationAnalytics {
  id: string;
  conversationId: string;
  userId: string;  // For aggregation, not individual tracking
  startedAt: Date;
  endedAt: Date | null;
  messageCount: number;
  userMessageCount: number;
  aiMessageCount: number;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  category: string;  // Derived from metadata flags
  outcome: 'completed' | 'abandoned' | 'error' | 'crisis_escalated';
  avgResponseTime: number;
  hadFallback: boolean;
}
```

### What NOT to Track

```typescript
// NEVER store these in analytics
interface PROHIBITED {
  messageContent: string;      // PHI
  userQuery: string;           // PHI
  aiResponse: string;          // PHI
  specificTopics: string[];    // Could reveal health info
  exactSentiment: 'sad';       // Could reveal mental state
}
```

## Implementation Pattern

### Tracking Conversation Start

```typescript
// src/lib/ai/analytics.ts
export async function trackConversationStart(
  conversationId: string,
  userId: string
): Promise<void> {
  await db.insert(conversationAnalytics).values({
    id: generateId(),
    conversationId,
    userId,
    startedAt: new Date(),
    messageCount: 0,
    totalTokens: 0,
    category: 'unknown',
    outcome: 'in_progress'
  });
}
```

### Tracking Message Exchange

```typescript
export async function trackMessageExchange(
  conversationId: string,
  tokens: { input: number; output: number },
  responseTimeMs: number,
  flags: { hadFallback: boolean; hasCrisisIndicator: boolean }
): Promise<void> {
  await db
    .update(conversationAnalytics)
    .set({
      messageCount: sql`message_count + 1`,
      totalTokens: sql`total_tokens + ${tokens.input + tokens.output}`,
      inputTokens: sql`input_tokens + ${tokens.input}`,
      outputTokens: sql`output_tokens + ${tokens.output}`,
      avgResponseTime: sql`(avg_response_time * (message_count - 1) + ${responseTimeMs}) / message_count`,
      hadFallback: flags.hadFallback,
      ...(flags.hasCrisisIndicator && { outcome: 'crisis_escalated' })
    })
    .where(eq(conversationAnalytics.conversationId, conversationId));
}
```

### Tracking Conversation End

```typescript
export async function trackConversationEnd(
  conversationId: string,
  outcome: 'completed' | 'abandoned' | 'error'
): Promise<void> {
  await db
    .update(conversationAnalytics)
    .set({
      endedAt: new Date(),
      outcome
    })
    .where(eq(conversationAnalytics.conversationId, conversationId));
}
```

## Category Detection (Metadata-Based)

Detect conversation categories WITHOUT reading content:

```typescript
// Categories based on metadata flags from AI response
interface AIResponseMetadata {
  usedCopingStrategies: boolean;
  usedCrisisProtocol: boolean;
  usedCheckInSupport: boolean;
  usedGeneralChat: boolean;
  requestedClarification: boolean;
}

function deriveCategory(metadata: AIResponseMetadata): string {
  if (metadata.usedCrisisProtocol) return 'crisis_support';
  if (metadata.usedCopingStrategies

... [truncated for markdown readability]

## Skill2Query (before)

- Use chatbot-analytics to Implement AI chatbot analytics and conversation monitoring. Use when adding conversation metrics, tracking AI usage, measuring user engagement with chat, or building conversation dashboards. Activates for AI analytics, token tracking, conversation categorization, and chat performance
- Can you Implement AI chatbot analytics and conversation monitoring. Use when adding conversation metrics, tracking AI usage, measuring user engagement with chat, or building conversation dashboards. Activates for AI analytics, token tracking, conversation categorization, and chat performance?
- I need help with Implement AI chatbot analytics and conversation monitoring. Use when adding conversation metrics, tracking AI usage, measuring user engagement with chat, or building conversation dashboards. Activates for AI analytics, token tracking, conversation categorization, and chat performance
