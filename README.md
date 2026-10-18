# UUID v7 Bench v3: CPU, CUDA RTX 5060 e colisões

CLI em Rust com executável CUDA separado para Windows x64. A versão v3 **mantém** `cpu`, `gpu`, `hybrid` e `probability` e adiciona o modo **`collision-test`**, que recebe UUIDs reais da CPU ou identificadores sintéticos v7 da GPU e **compara cada valor** em uma tabela de hashes em memória, inclusive entre lotes e dispositivos.

## Pré-requisitos

- Rust (toolchain `stable-x86_64-pc-windows-msvc`): https://rustup.rs/
- Visual Studio 2022 ou Build Tools com **Desktop development with C++**
- Para GPU: NVIDIA Driver + CUDA Toolkit 12.8+ (GPU RTX 5060, `sm_120`)
- Abra o **Developer PowerShell for VS 2022** (x64), de preferência, para `nvcc` localizar MSVC.

## Compilar

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build-windows.ps1

# Testes unitários Rust (não exigem GPU)
cargo test
```

O script compila `target\release\uuid-v7-bench.exe` e `target\release\uuid-gpu.exe` (este segundo apenas quando o CUDA estiver disponível). Você pode compilar apenas a CPU com `cargo build --release`.

## Executar

```powershell
# Medições de throughput originais (sem detectar colisões)
.\target\release\uuid-v7-bench.exe cpu --seconds 10 --threads 6
.\target\release\uuid-v7-bench.exe gpu --seconds 10
.\target\release\uuid-v7-bench.exe hybrid --seconds 10 --threads 6

# Teoria apenas, sem gerar UUIDs
.\target\release\uuid-v7-bench.exe probability --rate 145000000000 --seconds 1200

# NOVO: detecção de colisões reais nos UUIDs amostrados
.\target\release\uuid-v7-bench.exe collision-test --source cpu --samples 250000 --threads 6
.\target\release\uuid-v7-bench.exe collision-test --source gpu --samples 250000
.\target\release\uuid-v7-bench.exe collision-test --source hybrid --samples 250000 --threads 6

# Demonstração: reduzir artificialmente a entropia para produzir duplicações observáveis
.\target\release\uuid-v7-bench.exe collision-test --source hybrid --samples 250000 --entropy-bits 16

# Teste maior: 2 milhões de UUIDs por dispositivo (4 milhões no modo híbrido)
.\target\release\uuid-v7-bench.exe collision-test --source hybrid --samples 2000000 --threads 6
```

## Parâmetros de `collision-test`

| Opção | Padrão | Finalidade |
|---|---:|---|
| `--source` | `cpu` | `cpu`, `gpu` ou `hybrid` |
| `--samples` | `250000` | Quantidade de UUIDs **por dispositivo**; 1 a 2 milhões |
| `--threads` | Threads disponíveis | Threads CPU; preferível 6 ou 12 no Ryzen 5 4500 |
| `--batch` | `65536` | Quantidade de IDs por lote, 1 a 262144 |
| `--blocks` | `256` | Blocos CUDA, se GPU usada |
| `--tpb` | `256` | Threads CUDA por bloco, se GPU usada |
| `--entropy-bits` | `74` | `74` = identificadores sem alteração; `1..73` = descartar bits de entropia (simulação) |

**O teste é por quantidade, não por duração**: em `collision-test`, use `--samples`, não `--seconds`. Os modos originais de throughput ainda usam `--seconds`.

## Como a detecção funciona

1. **CPU** chama `Uuid::now_v7()` usando o crate oficial `uuid`.
2. **GPU** produz valores sintéticos no layout UUID v7 usando cuRAND Philox e timestamp amostrado **por lote**, não em cada ID. O CUDA envia valores binários pelo pipe para Rust; o Rust faz a análise. Esses valores **não são UUIDs v7 criptograficamente seguros para produção**.
3. Rust insere cada identificador completo de 128 bits em um `HashMap<u128, Producer>`; entradas repetidas são contadas. O conjunto **não é limpo entre lotes**, portanto detecta duplicações entre lotes e entre CPU/GPU.
4. O programa informa total, distintos, repetidos, repetidos entre dispositivos, valores inválidos e até cinco exemplos. A verificação está no lado CPU e pode se tornar o gargalo: o valor de UUIDs/s aqui mede **gerar + transferir + conferir**, não o throughput puro de CUDA.
5. `--entropy-bits N` conserva timestamp, versão e variante, mas **remove bits do campo aleatório propositalmente** para facilitar observar colisões. Esses conflitos são produzidos pela simulação, **não** são falhas da geração original de UUID v7.

### Limites e interpretação

- `--samples` é limitado a 2 milhões por dispositivo para conter uso de RAM; HashMap cresce proporcionalmente ao número de identificadores distintos. Mesmo poucos milhões podem consumir centenas de MB. Não armazena na GPU os milhões de UUIDs de uma vez; envia em lotes.
- **Zero colisões detectadas não garante impossibilidade de colisões.** Para UUID v7 com ~74 bits disponíveis e timestamps variados, não detectar nenhuma em milhões é o esperado.
- `Uuid::now_v7()` utiliza contador compartilhado e não equivale a 74 bits aleatórios independentes por milissegundo. A probabilidade teórica de `probability` é **modelo idealizado**, não risco medido de `now_v7()`.
- GPU Philox é pseudorrandômico, não CSPRNG. Timestamp por lote é adequado a este estudo de formato, mas não a aplicações de produção.
- A implementação foi preparada para compilação no Windows e não pôde ser executada neste ambiente, pois `rustc`, `cargo` e `nvcc` não estão instalados. Faça `cargo test` e compile no seu Windows.

## Fontes oficiais

- [RFC 9562, UUID v7](https://www.rfc-editor.org/rfc/rfc9562.html#section-5.7)
- [Rust uuid v7 / now_v7()](https://docs.rs/uuid/latest/uuid/struct.Uuid.html#method.now_v7)
- [NVIDIA CUDA GPUs, compute capability](https://developer.nvidia.com/cuda/gpus)
