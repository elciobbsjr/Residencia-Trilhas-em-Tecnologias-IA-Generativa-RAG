# Relatório — Avaliação de Estratégias de Chunking com LangChain

## 1. Objetivo

Comparar 10 estratégias de chunking em documentos convertidos de PDF para Markdown, analisando tamanho, quantidade, contexto, estrutura, overlap, embeddings e adequação para sistemas de RAG.

Foram processados **12 PDFs** com Docling. Os embeddings foram gerados com `openai/text-embedding-3-small`, com dimensão **1536**.

## 2. Pipeline

`PDF → Markdown → Chunking → Embeddings → JSON → Avaliação`

## 3. Estratégias avaliadas

| Teste | Estratégia | Configuração |
|---|---|---|
| 1 | fixed_200 | 200 caracteres sem overlap |
| 2 | fixed_500 | 500 caracteres sem overlap |
| 3 | fixed_1000 | 1000 caracteres sem overlap |
| 4 | fixed_2000 | 2000 caracteres sem overlap |
| 5 | fixed_overlap_50 | 500 caracteres com overlap 50 |
| 6 | fixed_overlap_200 | 500 caracteres com overlap 200 |
| 7 | paragraph | separação por parágrafos |
| 8 | three_sentences | 3 sentenças por chunk |
| 9 | recursive | Recursive 1000/100 |
| 10 | markdown_headers | separação por headings Markdown |

## 4. Conversão PDF → Markdown

| Documento | Caracteres | Headings | Linhas de tabela | Refs. imagem |
|---|---|---|---|---|
| attention_is_all_you_need | 48957 | 27 | 55 | 0 |
| bert_pretraining | 70235 | 33 | 84 | 0 |
| bioetica_e_ia | 51213 | 24 | 0 | 0 |
| escrita_academica_ia | 42678 | 20 | 12 | 0 |
| gpt3_language_models | 333822 | 57 | 454 | 0 |
| gpt4_technical_report | 289542 | 213 | 197 | 0 |
| instruct_gpt | 220009 | 132 | 299 | 0 |
| llama_foundation_models | 105322 | 54 | 236 | 0 |
| lora_low_rank_adaptation | 99729 | 40 | 183 | 0 |
| retrieval_augmented_generation | 71960 | 35 | 53 | 0 |
| scaling_laws_llm | 99742 | 52 | 100 | 0 |
| twitter_algoritmo | 54440 | 21 | 0 | 0 |

Os arquivos Markdown preservaram texto, headings e diversas tabelas. A análise automática não encontrou referências explícitas de imagens em `![...]` ou `<img>`. Assim, conteúdo exclusivamente visual não participou diretamente dos embeddings. Informações relacionadas a layout, figuras, gráficos, cores e relações espaciais podem ter sido perdidas.

## 5. Estatísticas dos 10 testes

| Teste | Estratégia | Chunks médios | Tam. médio | Mínimo | Máximo | Dimensão |
|---|---|---|---|---|---|---|
| 1 | fixed_200 | 247.33 | 198.26 | 13 | 200 | 1536 |
| 2 | fixed_500 | 99.33 | 494.04 | 63 | 500 | 1536 |
| 3 | fixed_1000 | 50 | 984.22 | 213 | 1000 | 1536 |
| 4 | fixed_2000 | 25.33 | 1950.88 | 440 | 2000 | 1536 |
| 5 | fixed_overlap_50 | 110 | 495.89 | 156 | 500 | 1536 |
| 6 | fixed_overlap_200 | 164.67 | 496.7 | 156 | 500 | 1536 |
| 7 | paragraph | 149 | 339.69 | 1 | 4291 | 1536 |
| 8 | three_sentences | 119 | 414.64 | 14 | 1834 | 1536 |
| 9 | recursive | 70.33 | 717.32 | 100 | 999 | 1536 |
| 10 | markdown_headers | 22.33 | 2222.37 | 10 | 9877 | 1536 |

A comparação direta utiliza os três documentos em que todos os dez testes foram executados.

## 6. Recuperação semântica

| Pos. | Teste | Estratégia | Top-1 | Hit@3 | MRR | Similaridade |
|---|---|---|---|---|---|---|
| 1 | 8 | three_sentences | 1 | 1 | 1.0 | 0.7319 |
| 2 | 3 | fixed_1000 | 1 | 1 | 1.0 | 0.7314 |
| 3 | 2 | fixed_500 | 1 | 1 | 1.0 | 0.7294 |
| 4 | 7 | paragraph | 1 | 1 | 1.0 | 0.7288 |
| 5 | 4 | fixed_2000 | 1 | 1 | 1.0 | 0.7278 |
| 6 | 9 | recursive | 1 | 1 | 1.0 | 0.7278 |
| 7 | 6 | fixed_overlap_200 | 1 | 1 | 1.0 | 0.7229 |
| 8 | 5 | fixed_overlap_50 | 1 | 1 | 1.0 | 0.715 |
| 9 | 1 | fixed_200 | 1 | 1 | 1.0 | 0.7108 |
| 10 | 10 | markdown_headers | 1 | 1 | 1.0 | 0.7049 |

Todas as estratégias empataram em Top-1, Hit@3 e MRR nas seis consultas. O maior valor médio de similaridade foi do **Teste 8 — three_sentences**, com **0.7319**.

# 7. Análise obrigatória

## 7.1 Qual estratégia gerou mais chunks?

**Teste 1 — fixed_200**, com média de **247.33 chunks**. Chunks menores aumentam a quantidade de vetores armazenados e pesquisados.

## 7.2 Qual gerou menos chunks?

**Teste 10 — markdown_headers**, com média de **22.33 chunks**. Unidades maiores reduzem o número de fragmentos.

## 7.3 Como o tamanho dos chunks variou?

Os testes fixos ficaram próximos de 200, 500, 1000 e 2000 caracteres. Estratégias baseadas em parágrafos, sentenças, Recursive e Markdown variaram conforme a estrutura natural do conteúdo.

## 7.4 Qual estratégia preservou melhor a estrutura dos documentos?

O **Teste 10 — Markdown Header Text Splitter** preservou melhor a hierarquia de headings e metadados de seção e subseção. Sua principal desvantagem foi produzir algumas seções grandes.

## 7.5 Como tabelas foram tratadas?

Diversas tabelas foram convertidas para Markdown usando `|`. O conteúdo textual das células foi preservado, mas estratégias de tamanho fixo podem dividir tabelas no meio e separar cabeçalhos de valores.

## 7.6 Como imagens foram tratadas?

Não foram encontradas referências explícitas de imagens nos Markdown analisados. Portanto, as imagens não participaram diretamente dos embeddings textuais.

## 7.7 Quais informações foram perdidas durante PDF → Markdown?

Principalmente informações visuais e de layout: imagens, gráficos, cores, posicionamento, diagramação e relações espaciais. Texto, headings e diversas tabelas foram preservados.

## 7.8 O chunking por caracteres fragmentou conceitos ou estruturas?

Sim. Cortes fixos podem ocorrer no meio de frases, parágrafos, referências e tabelas. O problema foi mais forte no Teste 1, com 200 caracteres.

## 7.9 O chunking por parágrafo produziu chunks muito grandes?

Em alguns casos, sim. A estratégia preserva a unidade semântica do parágrafo, mas não controla seu tamanho, e foram observados parágrafos com milhares de caracteres.

## 7.10 O chunking por sentença preservou melhor o contexto?

Sim. Três sentenças por chunk normalmente evitam cortes no meio de frases e apresentaram excelente recuperação semântica. No GPT-3 houve um caso excepcional acima do limite de tokens, tratado por subdivisão de segurança.

## 7.11 O Recursive Splitter apresentou vantagens?

Sim. O Recursive prioriza `parágrafos → linhas → espaços → caracteres`, oferecendo bom equilíbrio entre preservação do texto e controle do tamanho.

## 7.12 O Markdown Splitter preservou a estrutura semântica?

Sim. Ele associou os chunks aos headings e manteve metadados estruturais. Algumas seções ficaram grandes e exigiram controle adicional de tamanho.

## 7.13 Qual estratégia parece mais adequada para RAG?

O **Teste 9 — Recursive** é a principal recomendação geral por equilibrar contexto, estrutura e tamanho. O **Teste 8 — três sentenças** apresentou excelente resultado semântico, e o **Teste 3 — 1000 caracteres** foi mantido como baseline.

## 7.14 Quais estratégias devem ser descartadas?

Não seriam priorizadas: **Teste 1**, pela fragmentação excessiva; **Teste 6**, pelo overlap de 40% e redundância; **Teste 4**, pelos chunks grandes; e **Teste 10 sem limite adicional**, por poder gerar seções muito extensas.

## 7.15 Quais estratégias devem seguir nos próximos experimentos?

1. **Teste 9 — Recursive**; 2. **Teste 8 — três sentenças**; 3. **Teste 3 — 1000 caracteres**. As três foram aplicadas aos 12 documentos.

# 8. Conclusão

Chunks pequenos aumentaram a fragmentação e o número de embeddings, enquanto chunks muito grandes reduziram a granularidade. O chunking por três sentenças apresentou a maior similaridade na avaliação inicial, enquanto o Recursive apresentou o melhor equilíbrio operacional para documentos heterogêneos. Assim, o **Recursive foi selecionado como principal estratégia para RAG**, acompanhado de três sentenças e do baseline de 1000 caracteres.

# 9. Arquivos produzidos

- Markdown dos 12 PDFs
- chunks e embeddings
- metadados
- JSONs por documento e estratégia
- `summary.json`
- `RELATORIO.md`