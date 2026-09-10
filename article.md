# From GPT-2 to Kimi K3: The Journey of 22,580 Models in One

**Language:** English · [Русский](./article.ru.md)

*A comprehensive visual breakdown of how AI architecture evolved from simple scaling to intelligent memory management*

**By Ujjwal Balaji**

---

## The Number That Changes Everything

Let's start with a mind-bending fact:

**22,580**

That's how many GPT-2 models (from 2019) you could fit inside a single Kimi K3 model (from 2026).

We scaled up by a factor of 22,580 in just seven years.

But here's the question that keeps AI researchers up at night: **Is it just... scale?**

If you think throwing more parameters at the problem is the whole story, you're about to discover why that's only half the truth.

---



## What You'll Learn Today

This isn't just another "AI is getting bigger" article. We're going on a journey through the most fascinating architectural evolution in modern AI. By the end, you'll understand:

- Why your AI chatbot sometimes forgets what you said 10 minutes ago
- How we went from models that recompute everything to ones that remember intelligently
- What "attention" really means (and why it's not just a fancy word)
- Why the latest models are faster AND smarter, not just bigger

---



## Visual Interactive Breakdown

Before we dive deep into the technical details, I've created a **comprehensive visual interactive breakdown** of this entire paper that makes the concepts even easier to understand.

**[Explore the Interactive Visual Breakdown Here](https://programmerftw.github.io/llm-breakdown-/)**

This visual guide includes:

- Interactive architecture diagrams you can click and explore
- Animated explanations of attention mechanisms
- Side-by-side comparisons of different approaches
- Visual timelines of the evolution
- Playgrounds where you can adjust sequence lengths and see real-time effects
- Step-by-step visualizations of each mechanism

**This visual companion was designed to complement this article perfectly. I highly recommend viewing it alongside reading this piece!**

---



## Part 1: The GPT-2 Era - When We Were Simple

Let's rewind to 2019. GPT-2 was revolutionary, but looking back, it was almost *adorably* simple.

### The Basic Building Block

Imagine you're reading a book. With every word, your brain processes not just that word, but everything that came before it. That's essentially what GPT-2 does.

Here's the architecture in its simplest form:

Input → Token Embeddings + Position Embeddings → Transformer Block (repeat 12 times) → Language Model Head → Output Logits

Each transformer block contains:

- **Multi-head Self-Attention**: The "pay attention to relevant past words" mechanism
- **Feed-Forward Network**: The "process what you paid attention to" mechanism



### The Hidden Inefficiency Nobody Talked About

Here's where things get interesting. When GPT-2 generates text one word at a time (autoregressive decoding), it has a dirty little secret:

**It recomputes everything for every single new word.**

Think about that. When generating the 100th word of a response, GPT-2 is still computing representations for words 1 through 99, even though nothing about them has changed.

Without caching, generating the next token would require:

1. Process the entire input sequence again
2. Run through all attention calculations again
3. Only use the final position's output

This is like reading an entire book from the beginning every time you want to read one more sentence.

### The KV Cache Solution

The breakthrough was simple: **Store what you've already computed.**

When you process a token for the first time, you compute its Key (K) and Value (V) vectors. Instead of throwing them away, you keep them in memory.

This "KV cache" means:

- First token: Full computation
- Every subsequent token: Only compute for the new token, use cached values for everything else

The trade-off? Memory bandwidth becomes the bottleneck. That cache grows with sequence length, and for long conversations, it can become massive.

**GPT-2 Stats:**

- 124 million parameters
- 12 transformer blocks
- 12 attention heads
- Embedding dimension: 768
- ~50,000 possible tokens

Now multiply that by 22,580, and you get Kimi K3's 2.8 trillion parameters.

---



## Part 2: The Attention Problem - Why O(N²) Hurts



### Understanding the Quadratic Nightmare

Standard attention has a mathematical complexity of O(N²), where N is the sequence length.

What does this mean in practice?


| Sequence Length | Attention Operations |
| --------------- | -------------------- |
| 100 words       | 10,000               |
| 1,000 words     | 1,000,000            |
| 10,000 words    | 100,000,000          |
| 100,000 words   | 10,000,000,000       |


You can see the problem. Processing a full book? You're doing billions of operations just for attention.

### The Four Separate Costs of Attention

The cost of attention isn't just one number. It breaks down into four distinct costs that scale differently:

**1 Training Cost** Full N×N score matrices were materialized before FlashAttention (2020). This is where the O(N²) cost is most visible.

**2 Prefill Cost** The whole prompt is processed at once, requiring O(N²) score computation per layer.

**3 Decode Cost** Each step reads the whole KV cache: 2·N·D reads + 2·D writes to HBM. This is memory-bandwidth bound, not FLOP bound.

**4 KV Cache Cost** The cache grows O(N) and can become the memory-bandwidth bottleneck at long context.

### The Softmax Dependence

The culprit is the **softmax function**. Here's how attention works: Attention(Q,K,V) = softmax(Q × K^T / √d) × V

The problem is that `Q × K^T` creates a massive matrix where every query interacts with every key. You can't break this apart because softmax needs to see all the scores together.

### Flash Attention: The Stopgap

In 2020, Flash Attention arrived. It didn't change the mathematics but made it efficient enough that people stopped complaining about O(N²).

But the fundamental limitation remained: **As sequences get longer, computation grows quadratically.**

---



## Part 3: Linear Attention - Breaking the Quadratic Curse



### The Big Idea

What if we could make attention O(N) instead of O(N²)?

Linear attention does this through a clever trick:

Instead of softmax (which needs to see everything at once), it uses a **feature map** like ELU+1 that can be applied independently to each query and key.

**The Mathematical Breakthrough:**

Standard attention: Attention = softmax(Q × K^T) × V

Linear attention: Attention = φ(Q) × (φ(K)^T × V)

Where φ is a feature map (like ELU+1).

**Why this works:**

1. You can compute `(φ(K)^T × V)` once for all keys
2. Store this as a fixed-size state (D×D matrix)
3. For each new query, just multiply with this state



### What You Keep vs. What You Give Up


| Standard Attention          | Linear Attention                  |
| --------------------------- | --------------------------------- |
| N×N matrix                  | D×D matrix (D is fixed dimension) |
| Grows with sequence length  | Fixed size state                  |
| O(N²) complexity            | O(N) complexity                   |
| Can retrieve any past token | Information gets mixed together   |
| Every query sees every key  | Keys fold into fixed state        |
| KV cache grows with length  | O(1) decode memory                |




### The Trade-off Nobody Mentions

Linear attention is efficient, but at a cost: **it's less expressive.**

Softmax gives you a beautiful probability distribution over all past tokens. Linear attention with ELU+1? It's an approximation.

Think of it like this:

- Softmax is a high-resolution camera that captures every detail
- Linear attention is a summary: you get the gist, but lose some nuance

For many applications, this loss is acceptable. But for others? It can hurt performance.

**The State Problem:**

When you fold all KV pairs into a fixed D×D state, you lose the ability to retrieve individual past tokens. Everything gets mixed together.

This is like trying to have a conversation where you can only remember the general theme of what was said, not the specific details from 5 minutes ago.

---



## Part 4: DeltaNet - Smarter Memory Management



### The Overcapacity Problem

Linear attention has a hidden flaw: **the state eventually fills up.**

When you're using a fixed-size memory and keep adding to it without ever removing anything, you eventually hit capacity. After that, new information interferes with old information.

This is exactly what happens with linear attention's additive updates: Statenew = Stateold + k × v^T

You're always adding. Nothing ever leaves.

**The "Endlessly Adding" Problem**

As Ilya Schlag's paper "Fast Weight Programmers" eloquently puts it:

> *"When the sequence length exceeds storage capacity, the model may end up in an overcapacity regime. To properly operate under such a regime, the model should learn to dynamically interact with the memory contents and selectively decide which key-value associations to keep and which ones to delete. The purely additive instruction may be inappropriate for this purpose… endlessly adding new associations to a memory of finite size, as in Eq. 17, inevitably will reach a limit."*

This is the fundamental problem: adding without evicting leads to interference.

### DeltaNet's Solution: The Delta Rule

DeltaNet introduces a crucial insight: **When writing new information, first read what's already there, then only write what's different.** 1.Read: What does the current key retrieve from the cache?

2.Compute: Difference between what we want to write and what's already there

3.Write: Only the difference gets added

**Mathematically:** delta = value - (key @ stateold) statenew = stateold + key × delta

This is the **Delta Rule** from neural network theory, adapted for attention.

**Why it works better:**

- Old information is explicitly removed when overwritten
- New information is written precisely
- Less interference between associations



### The Delta Rule in Action

Let's trace through an example with real numbers:

1. **Write**: Key k₁ = 1, 0, Value v₁ = 2, 3
  - State becomes: k₁ × v₁ = 2, 3, 0, 0
2. **Write**: Key k₂ = 0, 1, Value v₂ = 4, 5
  - Read: What does k₂ retrieve? 0, 0
  - Delta: 4, 5 - 0, 0 = 4, 5
  - State: 2, 3, 4, 5
3. **Write**: Key k₁ again, Value v₁' = 6, 7
  - Read: What does k₁ retrieve? 2, 3
  - Delta: 6, 7 - 2, 3 = 4, 4
  - State: 2+4, 3+4, 4, 5 = 6, 7, 4, 5

See what happened? When we wrote to an existing key, we **updated** the information rather than just adding more on top.

### The Chunking Challenge

DeltaNet's theoretical improvement came with a practical problem: **it's hard to parallelize.**

In standard attention, you can compute all queries and keys at once with matrix multiplication. With DeltaNet, each update depends on the previous state, making sequential computation necessary.

**The Breakthrough: Chunked Computation**

The solution was to process inputs in chunks:

1. Split sequence into chunks of size C (often 64 or 128)
2. Within each chunk: Compute standard attention (this is parallelizable)
3. Between chunks: Update the state sequentially

**The trade-off:**

- Smaller chunks = less sequential work, but more overhead
- Larger chunks = more parallel work, but less efficient memory

This is where hardware comes in. GPUs love operations of size 64 or 128 because that's what their tensor cores are optimized for.

### The DPLR Formulation

The authors rewrote the delta updates to enable chunked computation:

**The reparameterized form:** Sₜ = Sₜ₋₁(I − βₜkₜkₜᵀ) + βₜvₜkₜᵀ

This formulation allows the chunked code to compute all deltas at once, enabling efficient parallel training.

---



## Part 5: Gated DeltaNet - Adding the Ability to Forget



### Why DeltaNet Still Has Problems

DeltaNet can update specific associations, but it still has a limitation: **it can't "forget" information on its own.**

If you have a conversation and switch topics completely, the previous topic's information is still in the state. You can only replace it by writing to specific keys, but what if the new information uses different keys?

**Real-world analogy:**

- DeltaNet is like having a filing cabinet where you can replace files
- But you can't throw away an entire drawer at once
- You have to know exactly which file to replace, one by one



### The Mamba Connection

Mamba (the state-space model) introduced a simple idea: **apply a decay factor to everything.** Statenew = α × Stateold + k × v^T

Where α is between 0 and 1.

When α < 1, all old information gradually decays. This is great for forgetting context from a long time ago.

**The problem with this approach:**

Everything decays equally. You can't say "forget what we were talking about 5 minutes ago but remember what we discussed at the beginning."

### Gated DeltaNet: The Best of Both Worlds

Gated DeltaNet combines:

1. **DeltaNet's precision**: Update specific associations
2. **Mamba's decay**: Ability to forget

The magic formula: state = α × state + delta × (1 - α)

When α = 1: Pure DeltaNet (no forgetting) When α = 0: Complete memory reset When 0 < α < 1: Controlled forgetting

**The key insight:** Alpha can be learned for each channel (dimension) independently, giving the model fine-grained control over what to remember and what to forget.

### Visual Comparison: DeltaNet vs Gated DeltaNet

**DeltaNet (α = 1):**

- State keeps growing
- Interference accumulates
- No capacity freeing
- After 6 writes: max |S| = 0.9

**Gated DeltaNet (α = 0.75):**

- State is bounded by the decay
- Capacity is freed
- Can reset during context switches
- After 6 writes: max |S| = 0.73

---



## Part 6: Kimi Delta Attention (KDA) - The Centerpiece



### The Per-Channel Breakthrough

KDA is the core of Kimi Linear: a gated delta-rule linear attention with **fine-grained, per-channel decay**.

Instead of one scalar α for everything: state = α × state + delta // Single scalar

We now have: state = Diag(αₜ) × state + delta // Per-channel decay

**Why this matters:**

Think of different channels as different aspects of understanding:

- Channel 1 might encode subject-verb relationships
- Channel 2 might encode temporal information
- Channel 3 might encode emotional tone

Each channel can now decide its own forgetting rate. Channel 1 might want to remember forever (α = 0.99), while Channel 3 might want to forget quickly (α = 0.8).

### The KDA State Update

The full mathematical formulation: Sₜ = Diag(αₜ) · Sₜ₋₁(I − βₜkₜkₜᵀ) + βₜvₜkₜᵀ

Where:

- **Diag(αₜ)**: Per-channel diagonal decay matrix
- **βₜ**: Learning rate for the delta update
- **kₜ, vₜ**: Key and value at time t
- **I**: Identity matrix



### The Reported Results

With an identical training recipe, Kimi Linear (48B total / 3B active, hybrid 3:1 KDA:MLA):

- **Outperforms full MLA attention** across evaluated tasks
- **Cuts KV cache by up to 75%**
- **Reaches up to 6× decode throughput** at 1M context

---



## Part 7: Hybrid Attention - Why Not One Mechanism?



### The Two-System Approach

Full attention retrieves any token exactly but pays a growing KV cache. Recurrent layers decode in O(1) but must compress. Kimi's answer is a **hybrid stack**:

- **Most layers run KDA**: Efficient, constant-size recurrent memory
- **Every fourth layer runs MLA**: Full softmax retrieval

**Why this works:**

**Global retrieval (MLA layers):**

- Keep compressed keys and values for every past token
- The model can still look anything up exactly
- Through the latent representation

**Efficient recurrence (KDA layers):**

- Carry a constant-size state through the whole sequence
- O(1) decode, no cache growth
- Information is evicted to fit the fixed state



### Multi-head Latent Attention (MLA) - Compressing the Cache

At long context, the KV cache itself becomes the bottleneck. MLA compresses keys and values into a shared latent vector per token and up-projects them on use.

**The Compression:**


| Aspect            | Traditional KV Cache | MLA Latent Cache     |
| ----------------- | -------------------- | -------------------- |
| Storage per token | Full K + V vectors   | Shared latent vector |
| Memory units      | 134.22 M             | 8.39 M               |
| Reduction         |                      | 94%                  |
| Reported          |                      | Up to 75% reduction  |


**How it works:**

1. Project K and V into a shared latent space
2. Store only the latent vector per token
3. On retrieval, up-project back to K and V
4. Trade exact storage for much smaller cache

---



## Part 8: Mixture of Experts (MoE) - Capacity Without Compute



### The Routing Problem

A learned router sends each token to a few experts, so total capacity can grow far beyond per-token compute.

**The Math:** y = Σ gₑ(x) · Eₑ(x) over top-k experts

**Kimi K3's MoE Configuration:**

- **898 total experts**
- **2 shared experts** (process every token)
- **896 routed experts**
- **16 selected per token** by the router



### The Scale Numbers


| Metric             | Value                    |
| ------------------ | ------------------------ |
| Total parameters   | 2.8 trillion             |
| Active per token   | ~104 billion             |
| Expert utilization | ~3.7% of capacity active |
| Router selection   | Top-16 of 896            |




### The Load Balancing Challenge

The router must balance load across experts during training:

- Otherwise, a few experts absorb most tokens
- The rest of the capacity is wasted
- Expert specialization becomes important



### Latent-Space Experts

In Kimi K3, experts operate in a compressed latent space:

1. Project input down to latent dimension
2. Run expert computation in latent space
3. Project back up to original dimension

**Result:** Expert computation is nearly twice as fast (halves the FLOPs).

### The SiTU Activation

Kimi K3 replaced the standard SiLU activation with **SiTU**: SiLU: x × sigmoid(x) SiTU: x × tanh(x) // Learnable β parameter

**Engineering Note:** Without a fused kernel, SiTU is almost 3× slower than the original path. But the latency-space MoE offset this by making the overall forward pass faster.

---



## Part 9: Attention Residuals (AttnRes) - Attending Over Depth



### The Residual Dilution Problem

In a standard residual stream, every layer's output is added with equal weight. This means:

- Each layer's relative share shrinks as the network deepens
- Later layers must learn ever-larger outputs to matter
- No selective access to earlier representations

**Standard residual:** hl = h₁ + Σ fᵢ(hᵢ) // Equal weighting for all

### The AttnRes Solution

AttnRes lets each layer retrieve from earlier representations instead of receiving one lossy sum: hl = α₀·h₁ + Σ αᵢ·fᵢ(hᵢ) where α = softmax(q·k)

Each weight αᵢ is computed from a query-key dot product over earlier residual states.

**The Benefits:**

**Standard** → Purely additive, no selective access, every layer sees the same aggregated state

**AttnRes** → Selective depth-wise retrieval, learned weights, each layer can focus on what matters

### Blockwise AttnRes in Kimi K3

- Applied every 12 layers
- Produces 8 AttnRes blocks across 23 macrocycles
- Adds roughly 2% inference latency
- Provides 1.25× compute advantage in scaling-law comparisons

**The Payoff:**

- Mitigates residual dilution
- Controls hidden-state growth
- Selective retrieval of earlier representations

---



## Part 10: The Kimi K3 Architecture - Everything Assembled



### The Full Picture

Kimi K3 combines all the innovations we've discussed:

- **2.8 trillion total parameters**
- **23 macrocycles** (each: 3×KDA + 1×MLA)
- **8 AttnRes blocks** (every 12 layers)
- **898 experts** (16 active per token)
- **1M token context**
- **~104B active parameters per token**



### Architecture Stack

For each macrocycle (23 total): Layer 1: KDA (Kimi Delta Attention) Layer 2: KDA (Kimi Delta Attention) Layer 3: KDA (Kimi Delta Attention) Layer 4: MLA (Multi-head Latent Attention)

MoE feed-forward layers

AttnRes every 12 layers (8 times total)

### The Three Pillars

**1 Recurrent Memory (KDA)**

- Constant-size state
- Per-channel decay
- Delta updates
- O(N) complexity

**2 Exact Retrieval (MLA)**

- Full softmax attention over context
- Periodic (every 4 layers)
- Resets and retrieves specific information
- Compressed KV cache (up to 75% reduction)

**3 Sparse Capacity (MoE)**

- 898 experts, 16 active per token
- Latent-space computation
- Massive capacity with sparse activation



### The AttnRes Glue

- Learned weighting of previous layers
- Applied every 12 layers
- Selective retrieval of depth-wise representations
- 1.25× compute advantage

---



## Part 11: The Architecture Evolution Map



### The Complete Lineage

Each node represents a wall hit and a fix purchased:

**GPT-2 (2019)**

- **Problem:** Decoding recomputes every position; attention couples every token
- **Idea:** Decoder-only Transformer
- **Benefit:** 124M params, general next-token predictor
- **Tradeoff:** O(N²) attention, KV cache grows O(N)

**Softmax Attention**

- **Problem:** How should tokens exchange information?
- **Idea:** Q·Kᵀ/√dₖ, causal mask, softmax, weighted values
- **Benefit:** Expressive, content-addressed retrieval
- **Tradeoff:** Softmax sits after q·k — every query couples to every key

**Linear Attention**

- **Problem:** The N² coupling, full score matrices materialized
- **Idea:** Feature map φ = elu+1 applied separately
- **Benefit:** O(N) cost, O(1) decode memory
- **Tradeoff:** φ approximates softmax; purely additive state → interference

**DeltaNet (NeurIPS 2024)**

- **Problem:** Additive memory overflows
- **Idea:** Read what's stored, write only the delta
- **Benefit:** Precise writes, hardware-efficient training
- **Tradeoff:** Can only overwrite with specific replacement

**Gated DeltaNet (Dec 2024)**

- **Problem:** No way to clear multiple associations
- **Idea:** Add Mamba-style gating
- **Benefit:** Bounded state, adaptive forgetting
- **Tradeoff:** Scalar decay is uniform

**KDA - Kimi Linear (Oct 2025)**

- **Problem:** Uniform decay ignores channel differences
- **Idea:** Per-channel diagonal gate, hybrid 3:1 KDA:MLA
- **Benefit:** Outperforms full attention, −75% KV cache
- **Tradeoff:** Complex kernels, careful hybrid design required

**MLA**

- **Problem:** KV cache dominates memory at long context
- **Idea:** Compress K and V into shared latent
- **Benefit:** Up to 75% KV cache reduction
- **Tradeoff:** Extra projections, compressed retrieval

**MoE**

- **Problem:** Capacity and compute were locked together
- **Idea:** Learned router sends tokens to top-k experts
- **Benefit:** 2.8T capacity, ~104B active per token
- **Tradeoff:** Routing, load balancing, expert specialization

**AttnRes**

- **Problem:** Residual dilution, equal weighting
- **Idea:** Each layer softmax-attends over earlier block outputs
- **Benefit:** Selective depth-wise retrieval, 1.25× compute advantage
- **Tradeoff:** +~2% inference latency

**Kimi K3 (Jul 2026)**

- **Problem:** Combine all approaches without one bottleneck
- **Idea:** 23 macrocycles of 3×KDA + 1×MLA, latent MoE, AttnRes
- **Benefit:** 2.8T params, 1M context, frontier results
- **Tradeoff:** Engineering complexity, fused kernels essential

---



## Part 12: The Bigger Picture



### What We've Learned

The journey from GPT-2 to Kimi K3 teaches us several important lessons:

**1 Scaling alone is not enough**

Yes, Kimi K3 has 22,580× more parameters than GPT-2. But the architectural innovations matter just as much:

- Linear attention (O(N) instead of O(N²))
- Delta updates (replace, don't just add)
- Gating (learn when to forget)
- Hybrid approaches (combine different memory types)

**2 Memory management is crucial**

The biggest challenge isn't computation—it's memory. How do you:

- Store information without it growing forever?
- Retrieve specific information when needed?
- Forget information that's no longer relevant?
- Balance between remembering everything and remembering nothing?

**3 Hardware drives architecture**

The chunk size in DeltaNet (64 or 128) isn't arbitrary—it's what GPUs are optimized for. The move to latent-space MoE was driven by the need to reduce FLOPs.

**4 Hybrid systems win**

No single approach is perfect:

- Pure attention is powerful but O(N²)
- Pure linear attention is efficient but loses detail
- Pure recurrent models have capacity limits

Combining them creates systems that are both powerful and efficient.

### The Final Insight

The most elegant part of this entire progression is how each innovation solves a specific problem created by the previous one:

1. **Attention** → O(N²) problem
2. **Linear Attention** → Fixed state, but interference problem
3. **DeltaNet** → Precise updates, but can't forget
4. **Gated DeltaNet** → Can forget, but uniform decay
5. **Kimi Linear** → Per-channel control, but still limited
6. **Kimi K3** → Everything together, with hardware optimization

Each step adds capacity, but only where it serves a specific functional role. The result is a system that's both powerful and efficient—22,580 GPT-2 models in one, but in a form that actually works.

### What's Next?

We're seeing the emergence of architectures that are:

- **Hardware-aware**: Designed for what GPUs and TPUs do well
- **Memory-efficient**: Handle long contexts without blowing up
- **Adaptive**: Learn what to remember and what to forget
- **Sparse**: Use massive capacity with minimal computation

The next generation might include:

- Even more sophisticated forgetting mechanisms
- Better ways to combine different memory types
- Hardware specifically designed for these architectures
- Models that can handle infinite context length

---



## Summary: The Technical Evolution

For those who want the technical breakdown:


| Model            | Year | Key Innovation                   | Memory Type             | Complexity             |
| ---------------- | ---- | -------------------------------- | ----------------------- | ---------------------- |
| GPT-2            | 2019 | KV Cache, Standard Attention     | Growing with length     | O(N²)                  |
| Linear Attention | 2020 | Feature maps, Fixed state        | Fixed size D×D          | O(N)                   |
| DeltaNet         | 2022 | Delta rule, Precise updates      | Fixed size with updates | O(N) but sequential    |
| Gated DeltaNet   | 2023 | Learnable decay, Forgetting      | Fixed size with decay   | O(N)                   |
| Kimi Linear      | 2025 | Per-channel decay, Hybrid        | Hybrid (KDA + MLA)      | O(N) + periodic O(N²)  |
| Kimi K3          | 2026 | All of the above + MoE + AttnRes | Multi-level memory      | Optimized for hardware |




### The Architecture Breakdown

**KDA (Kimi Delta Attention):**

- Constantly sized recurrent memory
- Per-channel decay (learned)
- Delta updates (precise overwriting)
- O(N) complexity

**MLA (Multi-head Latent Attention):**

- Full softmax attention over context
- Periodic (every 4 layers)
- Resets and retrieves specific information
- O(N²) but only when needed

**MoE (Mixture of Experts):**

- 898 experts, 16 active per token
- Latent-space computation (compressed)
- Massive capacity with sparse activation

**AttnRes (Attention Residuals):**

- Learned weighting of previous layers
- Applied every 12 layers (8 times total)
- Selective retrieval of depth-wise representations

---



## Final Thoughts

The journey from GPT-2 to Kimi K3 shows us something profound about AI development: **progress isn't just about making things bigger. It's about making things smarter.**

Each architectural innovation addressed a concrete limitation. The result is not just a bigger model, but a better one—one that handles memory intelligently, retrieves information selectively, and knows when to forget.

As we look to the future, we should expect more of the same: architectures that are designed with hardware in mind, that combine different approaches to solve different problems, and that ultimately create systems that are more capable than the sum of their parts.

The journey from 22,580 GPT-2 models to one Kimi K3 model is a journey from simple scaling to intelligent design. And that's the real story of AI progress.

---



## Interactive Visual Companion

Don't forget to explore my **interactive visual breakdown** of this paper:

**👉** **[Explore the Interactive Visual Breakdown Here](https://programmerftw.github.io/llm-breakdown-/)**

This visual guide is designed to complement the article with:

- Interactive diagrams showing each architecture
- Clickable components that explain each mechanism
- Animated explanations of attention operations
- Side-by-side comparisons
- Visual timelines of the evolution
- Playgrounds where you can adjust sequence lengths and see real-time effects
- Step-by-step visualizations of each mechanism
- Mathematical playgrounds with real number examples

---



## References

This article was created based on the detailed technical worklog by **@waterloointern** on X. The original worklog provided the foundation for understanding these architectural evolutions.

**Original Source:** [@waterloointern's X Post](https://x.com/waterloo_intern/status/2081762065392541951)

**Visual Companion:** [Interactive Visual Breakdown](https://programmerftw.github.io/llm-breakdown-/) by Ujjwal Balaji

For those who want to dive deeper into the academic literature:

1. **GPT-2**: "Language Models are Unsupervised Multitask Learners" (2019) - Radford et al.
2. **Linear Attention**: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020) - Katharopoulos et al.
3. **DeltaNet**: "Parallelizing Linear Transformers with the Delta Rule" (2022) - Schlag et al.
4. **Gated DeltaNet**: Combined insights from DeltaNet and Mamba architectures
5. **Kimi Linear**: Technical reports and papers from the Kimi team
6. **Kimi K3**: The architecture described in the original worklog
7. **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) - Gu and Dao
8. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) - Dao et al.
9. **Fast Weight Programmers**: "Fast Weight Programmers" - Schlag et al.

---



## 📚 Article Reference

**Title:** From GPT-2 to Kimi K3: The Journey of 22,580 Models in One

**Author:** Ujjwal Balaji

**Based on:** Technical worklog by @waterloointern on X

**Visual Companion:** [Interactive Visual Breakdown](https://programmerftw.github.io/llm-breakdown-/)

---

*Written for Hashnode by* ***Ujjwal Balaji***

*Feel free to share this article with anyone interested in the evolution of AI architecture. The goal is to make these complex topics accessible to everyone, from curious beginners to experienced researchers.*
