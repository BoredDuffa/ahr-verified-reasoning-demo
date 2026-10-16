import Lake
open Lake DSL

package lean_ahr_mathlib where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.30.0"

@[default_target]
lean_lib LeanAhrMathlib where
