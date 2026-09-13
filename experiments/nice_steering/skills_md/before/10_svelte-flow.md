# 10. svelte-flow (before)

- skill_id: `22d0b096-b677-4f9a-9b18-5c469f693e75`
- target_token: `NICE`
- token_propagation_before: `0.0000`

## Before Skill

- skill_id: `22d0b096-b677-4f9a-9b18-5c469f693e75`
- name: `svelte-flow`
- description: Build node-based editors, interactive diagrams, and flow visualizations using Svelte Flow. Use when creating workflow editors, data flow diagrams, organizational charts, mindmaps, process visualizations, DAG editors, or any interactive node-graph UI. Supports custom nodes/edges, layouts (dagre, hierarchical), animations, and advanced features like proximity connect, floating edges, and contextual zoom.
- tags: []

### capabilities

- `cap_1`: Build node-based editors, interactive diagrams, and flow visualizations using Svelte Flow. Use when creating workflow editors, data flow diagrams, organizational charts, mindmaps, process visualizations, DAG editors, or any interactive node-graph UI. Supports custom nodes/edges, layouts (dagre, hierarchical), animations, and advanced features like proximity connect, floating edges, and contextual zoom.

### parameters

- (none)

### examples

- (none)

### body

---
name: svelte-flow
description: Build node-based editors, interactive diagrams, and flow visualizations using Svelte Flow. Use when creating workflow editors, data flow diagrams, organizational charts, mindmaps, process visualizations, DAG editors, or any interactive node-graph UI. Supports custom nodes/edges, layouts (dagre, hierarchical), animations, and advanced features like proximity connect, floating edges, and contextual zoom.
keywords: [svelte-flow, visualization, nodes, graph-editor, diagrams, interactive-ui]
---

# Svelte Flow

Expert guide for building node-based UIs with Svelte Flow (@xyflow/svelte).

## Installation

```bash
npm install @xyflow/svelte
```

## Core Setup

```svelte
<script lang="ts">
  import { SvelteFlow, Background } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  
  // Use $state.raw for nodes and edges (Svelte 5)
  let nodes = $state.raw([
    { id: '1', data: { label: 'Node 1' }, position: { x: 0, y: 0 } },
    { id: '2', data: { label: 'Node 2' }, position: { x: 0, y: 100 } }
  ]);
  
  let edges = $state.raw([
    { id: 'e1-2', source: '1', target: '2' }
  ]);
</script>

<SvelteFlow bind:nodes bind:edges fitView>
  <Background />
</SvelteFlow>
```

## Key Concepts

### Nodes

Each node requires:
- `id`: Unique identifier
- `position`: `{ x: number, y: number }`
- `data`: Object with custom properties (typically includes `label`)
- Optional: `type`, `style`, `hidden`, `width`, `height`, `sourcePosition`, `targetPosition`

```typescript
const node = {
  id: '1',
  type: 'custom',
  position: { x: 100, y: 100 },
  data: { label: 'My Node', color: '#ff0000' },
  style: 'background: #f0f0f0',
  width: 150,
  height: 100
};
```

### Edges

Each edge requires:
- `id`: Unique identifier
- `source`: Source node ID
- `target`: Target node ID
- Optional: `type`, `label`, `animated`, `markerEnd`, `markerStart`, `style`, `hidden`

```typescript
const edge = {
  id: 'e1-2',
  source: '1',
  target: '2',
  type: 'smoothstep',
  label: 'connects to',
  animated: true,
  markerEnd: { type: MarkerType.ArrowClosed }
};
```

### Built-in Types

**Edge Types:**
- `default`: Straight line
- `smoothstep`: Smooth 90-degree turns
- `step`: Hard 90-degree turns
- `straight`: Alias for default
- `bezier`: Curved bezier line

**Node Types:**
- `default`: Basic rectangle node
- `input`: Node with only source handles
- `output`: Node with only target handles

## Custom Nodes

```svelte
<!-- CustomNode.svelte -->
<script lang="ts" module>
  import type { Node, NodeProps } from '@xyflow/svelte';
  
  export type CustomNodeType = Node<{
    label: string;
    color?: string;
  }>;
</script>

<script lang="ts">
  import { Handle, Position } from '@xyflow/svelte';
  
  let { data }: NodeProps<CustomNodeType> = $props();
</script>

<div class="custom-node" style:background={data.color}>
  <Handle type="target" position={Position.Left} />
  <div>{data.label}</div>
  <Handle type="source" position={Position.Right} />
</div>

<style>
  .custom-node {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ddd;
  }
</style>
```

Register custom nodes:

```svelte
<script lang="ts">
  import CustomNode from './CustomNode.svelte';
  
  const nodeTypes = {
    custom: CustomNode
  };
  
  let nodes = $state.raw([
    { id: '1', type: 'custom', data: { label: 'Custom', color: '#ff7000' }, position: { x: 0, y: 0 } }
  ]);
</script>

<SvelteFlow bind:nodes {nodeTypes} bind:edges fitView />
```

## Custom Edges

```svelte
<!-- CustomEdge.svelte -->
<script lang="ts">
  import { BaseEdge, EdgeLabel, getStraightPath, type EdgeProps } from '@xyflow/svelte';
  
  let { id, sourceX, sourceY, targetX, targetY, label }: EdgeProps = $props();
  
  let [edgePath, labelX, labelY] = $derived(
    getStraightPath({ sourceX, sourceY, targetX, targetY })
  );
</script>

<BaseEdge {id} path={edgePath} />
{#if label}
  <EdgeLabel x={labelX} y={labelY}>
    <div class="edge-label">{label}</div>
  </EdgeLabel>
{/if}
```

Path helpers available:
- `getStraightPath()`
- `getBezierPath()`
- `getSmoothStepPath()`

Register custom edges:

```svelte
<script lang="ts">
  const edgeTypes = {
    custom: CustomEdge
  };
</script>

<SvelteFlow bind:nodes bind:edges {edgeTypes} fitView />
```

## Updating State

**Critical:** Nodes and edges are immutable. Create new objects to trigger updates.

```svelte
<script lang="ts">
  // ❌ Won't work - direct mutation
  nodes[0].position.x = 100;
  
  // ✅ Works - create new array
  nodes = nodes.map((node) => {
    if (node.id === '1') {
      return { ...node, position: { ...node.position, x: 100 } };
    }
    return node;
  });
  
  // ✅ Also works - using helper from useSvelteFlow
  import { useSvelteFlow } from '@xyflow/svelte';
  const { updateNode } = useSvelteFlow();
  
  updateNode('1', (node) => ({
    ...node,
    position: { ...node.position, x: 100 }
  }));
</script>
```

## Event Handling

Common event handlers:

```svelte
<SvelteFlow
  bind:nodes
  bind:edges

... [truncated for markdown readability]

## Skill2Query (before)

- Use svelte-flow to Build node-based editors, interactive diagrams, and flow visualizations using Svelte Flow. Use when creating workflow editors, data flow diagrams, organizational charts, mindmaps, process visualizations, DAG editors, or any interactive node-graph UI. Supports custom nodes/edges, layouts (dagre, hierarchical), animations, and advanced features like proximity connect, floating edges, and contextual zoom
- Can you Build node-based editors, interactive diagrams, and flow visualizations using Svelte Flow. Use when creating workflow editors, data flow diagrams, organizational charts, mindmaps, process visualizations, DAG editors, or any interactive node-graph UI. Supports custom nodes/edges, layouts (dagre, hierarchical), animations, and advanced features like proximity connect, floating edges, and contextual zoom?
- I need help with Build node-based editors, interactive diagrams, and flow visualizations using Svelte Flow. Use when creating workflow editors, data flow diagrams, organizational charts, mindmaps, process visualizations, DAG editors, or any interactive node-graph UI. Supports custom nodes/edges, layouts (dagre, hierarchical), animations, and advanced features like proximity connect, floating edges, and contextual zoom
