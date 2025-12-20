## sl:0

Good afternoon dear comiittee. My name is.. . Today I am going present annual progress on my thesis: Machine learning search for perspective catalytic alloys

## sl:1

My academic progress this year consist of TAship and Philosophy exam
I also participated in two conferences and published proceedings there.

## sl:2: Catalysts that Changed the World

Throughout history, catalysts have transformed daily life and global industries. Here are three well-known examples:

- **Haber–Bosch reaction** — used to make ammonia for fertilizers. This process enabled large-scale food production and helped support the world’s population growth.
- **Catalytic converters** — platinum-based catalysts that clean car exhaust, removing poisonous gases and reducing air pollution.
- **Hydrogenation reactions** — essential for producing margarine, plastics, and many medicines, forming the backbone of the chemical industry.

All of these breakthroughs became practical and economically viable because of catalysts.

In my PhD work, I focus on the **Hydrogen Evolution Reaction**, which generates hydrogen as a clean fuel.
This reaction is especially useful in research, because it has a simple experimental setup and serves as a reliable benchmark for testing new catalysts.

> So catalysts have shaped modern society in countless ways.

## sl:3: Problem statement :: Catalyst Discovery

But today, we face a problem: many of our most important catalysts still rely on scarce and expensive metals, and finding affordable alternatives at scale remains a major challenge.

Why it matters:
This reliance on costly metals makes technologies more expensive and less sustainable.
If we can discover catalysts made from abundant and affordable elements, it will open the door to cleaner and more widely accessible technologies.

> This sets the stage for exploring how computational methods can help us search for alternatives more effectively.

## sl:4: Metrics :: Cost vs Performance

Here is an example of how we measure catalyst performance.
The metric is hydrogen peroxide productivity in the oxygen reduction reaction.

If we compare palladium with a nickel–platinum alloy:

- Per gram, their productivity is similar.
- Per dollar, the nickel–platinum alloy is about ten times more effective.

The key point is that cost matters as much as performance.
[Cheaper alloys can actually] outperform pure precious metals once you take price into account.

> This is why we always need to balance both cost and performance when evaluating catalysts.

## sl:5: Goals && Objectives

Our main goal is to test whether Graph Neural Networks can be used for large-scale screening of transition-metal alloys, specifically for bimetallic catalysts.
We also want to identify which models are reliable enough for practical use.

To reach this goal, we set four objectives:

1. Build a benchmark dataset for systematic testing.
2. Compare GNN predictions against Density Functional Theory, or VASP, on alloy datasets.
3. Identify the top-performing GNN models for large-scale screening.
4. Finally, pick the most promising alloy candidates for experimental validation.

> Now that we’ve set the goal and objectives, let’s walk through the screening protocol we use to evaluate catalyst candidates.

## sl:6: Methods :: Screening Protocol

We start with 30 transition metals and 10 prototype structures. For each prototype, we combine every possible pair of metals. In total, that gives us more than five thousand candidate alloys.

To narrow them down, we apply three main filters:

1. **Thermodynamic stability** — we check the formation energy.
   Lower values mean the material is more stable.
2. **Catalytic activity** — we compare the density of states, or DOS, of each alloy to a known reference catalyst, palladium.
   This tells us how similar the electronic behavior of the alloy is to palladium.

After applying these steps, we select the most promising candidates for experimental testing.
[This way, we reduce] thousands of possibilities to just a handful worth pursuing in the lab.

> To apply this screening funnel, we need reliable calculations and scalable ways to calculate properties. For that, we compare two main approaches: DFT and Graph Neural Networks

## sl:7: Methods :: Comparison

VASP, which is based on Density Functional Theory, solves quantum equations from first principles.
It’s our ground truth benchmark — very accurate, but also very slow.
Running VASP requires high-performance computing clusters.

In contrast, Graph Neural Networks don’t solve physics directly.
They learn patterns of energy and stability from existing DFT-calculated data.
That makes them less precise, but much faster and highly scalable.

Here is the difference in throughput:

- With VASP on a cluster, we can calculate about 1,500 structures per day.
- With a GNN, running on a laptop, we can handle over 70,000 structures per day.

So GNNs are roughly 50 times faster than VASP.
This speed makes it possible to explore huge datasets, while VASP remains the accuracy benchmark.
Together, they form an efficient discovery pipeline.

[This balance] of speed and accuracy is what makes GNNs so powerful for catalyst discovery.

> To test both VASP and GNNs fairly, we need well-defined datasets. Let me show you the three datasets we use for benchmarking.

## sl:8: Methods :: Datasets for Graph Neural Network benchmarking

<!-- We use three different datasets to benchmark Graph Neural Networks. -->

1. **Binary alloys** — about 5,200 structures.
   These are pairs of transition metals in prototype crystal structures.
   These is our target dataset. Resulting alloys are highly relevant, since many of them can actually be synthesized.

2. **Random 16-atom structures** — about 5,000.
   These are random arrangements of transition metals.
   They help explore the composition space, though they are less realistic.

3. **Liquid 64-atom metals** — about 3,000.
   These are unconventional systems, such as Al, Na, Au, Cu, and N clusters.
   They are not meant for synthesis, but they are a good stress test for GNN performance.

So across these three datasets, we cover practical alloys, exploratory cases, and challenging edge cases.

[This diversity] of datasets lets us test how well GNNs generalize beyond the training distribution.

> Now that we have our datasets, let’s see how different Graph Neural Networks perform on the binary alloys

## sl:9: Results :: Binary Alloys :: Graph Neural Networks vary - best vs worst performance

Here we compare two GNN models on the binary alloy dataset.

On the left is **ESEN**, our worst-performing model.
It shows a large scatter compared to the VASP reference.
The error is about 0.34 electronvolts per atom, with a low R² of 0.41.

On the right is **SevenNet**, our best model.
Here the predictions align much more closely with VASP.
The error drops to about 0.09 electronvolts per atom, and R² improves to 0.96.

The key point is that models that look similar on standard benchmarks can behave very differently on new datasets.
So model selection really matters.

[This shows] why it’s important to test models beyond standard benchmarks.

> So we’ve seen that GNN models can vary a lot in quality. Let’s now look at how they actually perform when used for screening stable versus unstable structures.

## sl:10: Results :: Scatter plots

Here we test how well different GNN models classify alloys as stable or unstable.

Most alloys cluster near zero formation energy in VASP, meaning they are slightly stable or metastable.

The table shows performance across models:

- The best models, **SevenNet and MACE-MPA**, have the lowest error and correctly classify most stable structures.
- The worst model, **ESEN**, has much larger errors and misclassifies many unstable structures as stable.

Overall, GNNs tend to underestimate formation energy, which makes them less aggressive filters than VASP.
That means they pass more candidates through, which is actually better than less candidates comparing to VASP

The takeaway is that model choice strongly affects accuracy.
This is why benchmarking against VASP is essential before using GNNs in discovery pipelines.

> But performance is not the same for every chemical system. Let’s look at how prediction errors vary depending on which elements are involved.

## sl:10: Results :: Heatmap

This heatmap shows how well SevenNet matches VASP for different binary alloys.
Each square represents the mean error for a pair of elements.
Bright colors mean good agreement, while dark colors mean larger errors.

We see that alloys containing **niobium, hafnium, and chromium** show the largest deviations.
On the other hand, **platinum-based alloys** are predicted quite reliably.

The key point is that model accuracy is not uniform.
It depends strongly on which chemical system you are studying.
So even the best GNNs may perform very well in some regions of chemistry, and worse in others.

> So even within binary alloys, accuracy depends on which elements are involved. But what happens if we push the models with completely random structures?

## sl:11: Results :: Heatmap

Here we test the models on random 16-atom structures.
These are far from equilibrium and many of them are physically unrealistic.

On this dataset, all GNNs perform very poorly.
The mean error is around 0.6 electronvolts per atom, and the correlation with VASP is essentially zero.
The scatter plots for ESEN and SevenNet show the same story — no clear alignment with VASP.

The histogram on the right confirms that most of these random structures are high in energy and unstable.

The key point is that models trained on realistic data cannot generalize to stochastic inputs.
This shows the limits of GNNs outside their training distribution.

> So while GNNs work well for realistic alloys, they completely fail on random structures. Let’s wrap up with the main conclusions.

## sl:12: Conclusions

To summarize:

- Graph Neural Networks are suitable for high-throughput screening of bimetallic catalysts.
  Among the models we tested, **SevenNet performed best**.

- However, GNNs cannot be treated as universal interatomic potentials.
  They fail when applied to random / chaotic structures.

- Finally, we developed a comprehensive benchmark dataset to evaluate these models.
  A paper describing this dataset is currently in preparation.

> We’ve shown what GNNs can and cannot do for catalyst discovery. Now let me outline the next steps in this project.

## sl:13: Future

Looking ahead, we plan four main steps:

1. **Publish article** — finalize and submit the manuscript on our GNN benchmarking dataset.
2. **Refine the candidate set** — filter the ~1,445 alloys further, using density of states similarity.
3. **Synthesis** — prepare the top-ranked alloy candidates in the lab.
4. **Experimental validation** — test their catalytic activity in the hydrogen evolution reaction.

These steps will connect computational predictions with experiments, moving us closer to practical catalyst discovery.

<!-- To conclude, Graph Neural Networks offer a powerful way to speed up catalyst discovery, but they are not universal solutions. By combining them with reliable benchmarks and experimental validation, we can move from thousands of theoretical candidates to practical materials that support core industrial reactions. -->
