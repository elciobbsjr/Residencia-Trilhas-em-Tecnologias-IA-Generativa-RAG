# Relatório - Aula 05

## Exercício 1 - Criando Documents

Foram criados **6 objetos `Document`** sobre embeddings, chunking, RAG, tokenização e metadados.

O resultado de `len(documentos)` foi:

```text
6
```

Cada `Document` possui `page_content`, responsável pelo texto, e `metadata`, responsável pelas informações associadas.

### Que tipos de dados são aceitos em `metadata`?

Nos testes realizados, `metadata` aceitou strings, números, valores booleanos, listas e dicionários aninhados.
A lista utilizada em `tags` e o dicionário utilizado em `detalhes` foram armazenados normalmente.
É importante observar que uma vector store pode possuir restrições próprias para os tipos de metadados utilizados em operações de filtragem.

### O que acontece se um `Document` for criado sem `metadata`?

Não ocorre erro. Quando `metadata` não é informado, o LangChain utiliza automaticamente um dicionário vazio (`{}`).

## Exercício 2 - Schema de metadados

| Campo | Tipo | Descrição |
|---|---|---|
| `fonte` | string | Nome do arquivo .md de origem |
| `documento_id` | string | Identificador do documento |
| `chunk_index` | int | Posição do chunk dentro do documento |
| `estrategia` | string | Estratégia de chunking utilizada |
| `chunk_size` | int/null | Tamanho configurado para o chunk |
| `chunk_overlap` | int | Overlap configurado |
| `n_caracteres` | int | Quantidade real de caracteres |
| `chunk_id` | string | Identificador único do chunk |
| `pagina` | int/null | Página original, quando disponível |
| `secao` | string/null | Seção ou heading associado |
| `n_palavras` | int | Quantidade aproximada de palavras |

### Campos próprios adicionados

- **`chunk_id`**: permite identificar unicamente o trecho recuperado.  
  **Pergunta que permite responder:** "Qual chunk específico foi utilizado para recuperar esta informação?"

- **`pagina`**: permite localizar a informação no documento original.  
  **Pergunta que permite responder:** "Em qual página do documento está essa informação?"

- **`secao`**: permite identificar a parte estrutural do documento à qual o trecho pertence.  
  **Pergunta que permite responder:** "Em qual seção do documento esse conteúdo aparece?"

- **`n_palavras`**: permite analisar o tamanho textual dos chunks.  
  **Pergunta que permite responder:** "Quantas palavras possui esse chunk e qual é sua granularidade em relação aos demais?"

### Exemplo preenchido com um chunk real

O exemplo abaixo foi obtido do documento `bioetica_e_ia`, utilizando a estratégia Recursive da Aula 04.

```json
{
  "page_content": "No entanto, a rápida expansão e incorporação da IA à prática clínica não ocorre sem desafios. A  adoção  acelerada  dessas  tecnologias  suscita preocupações éticas significativas, especialmente no que diz respeito aos princípios fundamentais da bioética. Desde seus primórdios, a medicina tem sido regida por princípios éticos que visam à proteção e ao bem-estar do paciente. O Juramento de Hipócrates, datado do século V a.C., consagrou valores  fundamentais  como  beneficência,  não maleficência, confidencialidade e respeito à auto -nomia do doente - pilares da bioética ainda hoje 4 . Tais valores foram reforçados no século XX por documentos como o Código de Nüremberg e  a Declaração de Helsinki , que consolidaram as bases para a ética em pesquisa e prática médica 3 .",
  "metadata": {
    "fonte": "bioetica_e_ia.md",
    "documento_id": "bioetica_e_ia",
    "chunk_index": 10,
    "estrategia": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "n_caracteres": 777,
    "chunk_id": "bioetica_e_ia_test09_chunk0011",
    "pagina": null,
    "secao": null,
    "n_palavras": 121
  }
}
```

Os campos `pagina` e `secao` aparecem como `null` neste exemplo porque essas informações não estavam disponíveis nos metadados gerados na etapa anterior.

### Qual campo incluir para citar a fonte na resposta final do RAG?

O principal campo adicional seria `pagina`, utilizado em conjunto com `fonte`. O campo `secao` também pode complementar a referência quando estiver disponível.

Assim, uma resposta poderia apresentar uma referência como: **Fonte: bioetica_e_ia.md, página 10, seção Discussão.**

No exemplo atual, `pagina` e `secao` estão como `null`, pois esses dados não foram preservados nos metadados da etapa anterior. Para uma citação exata por página em um sistema RAG futuro, essas informações precisariam ser mantidas durante a extração e o processamento do documento.

### Por que `chunk_index` é útil?

O `chunk_index` registra a posição do chunk dentro do documento.
Se um trecho recuperado estiver cortado no meio de uma explicação, é possível buscar o chunk anterior (`chunk_index - 1`) ou o posterior (`chunk_index + 1`) para recuperar mais contexto.
Além disso, o campo ajuda a reconstruir a ordem original dos chunks.

## Conclusão

A classe `Document` padroniza a representação dos conteúdos utilizados pelo LangChain. O texto fica em `page_content`, enquanto as informações necessárias para rastreamento, filtragem e citação ficam em `metadata`.
Os embeddings não são armazenados dentro do `Document`, pois são responsabilidade da vector store.