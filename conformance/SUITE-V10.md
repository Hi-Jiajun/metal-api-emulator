# v10: disjoint views of one allocation

The v10 suite is the first conformance dimension that exercises **ranged
aliasing**: two buffers of one case name the *same* allocation while their
byte ranges stay disjoint. It is the external, five-path counterpart of
`research/docs/14` steps 2 and 4, and it is the dimension that was blocked
until a provider declared `AliasMode::DistinctViews` (`8aaad1a`).

## Cases

| Case | Entry | Allocation | Views |
|---|---|---|---|
| `alias_disjoint_pair` | `copy_word` | 500 (16 bytes) | read at 4, write at 8 |
| `alias_disjoint_pair_reversed` | `copy_word` | 500 (16 bytes) | read at 8, write at 4 |

Both cases are single-pass copies of one 32-bit word. The reversed case binds
the source *above* the destination so an offset mix-up cannot pass. Every view
keeps at least four guard bytes before and after it, and the guard bytes belong
to the shared allocation, so a wrong offset or extent changes the observed
allocation image.

The suite deliberately reuses the reviewed `copy_word` fixture and the same
grid and local size as v1's `copy_word` case. It adds no new shader.

## What the comparator enforces

- Each allocation appears at most once in a capture's `allocations` list and
  is compared as one extent, so the two views must agree on a single image.
- Initialization ranges of one allocation must stay disjoint: overlapping
  views would make the observed bytes depend on write order and are refused
  with `overlapping initialization would depend on write order`.
- Several buffers naming one allocation are only qualified by this suite; a v1
  through v9 fixture that shares an allocation is refused by name.

## What this suite does not claim

- It does not admit overlapping views of one allocation. Every overlap,
  including read-read, is still refused by provider admission.
- It does not claim concurrent execution. The object API still reserves the
  whole allocation for the commit-to-completion window, so two commands that
  touch disjoint ranges of one allocation serialize. That is `research/docs/14`
  step 3.
- It is not a general Metal aliasing claim: the fixtures are bounded copies on
  the reviewed `copy_word` program.
