Logs of the 2026-10-31 g6.4xlarge session (NVIDIA L4, AMD EPYC 7R13, 16 vCPUs, us-east-1).

Builds: main = 7edea45; "br" (Phase B) = feat/444-10bit f8874bd; "fix" = cc87534 (k_recon split into one
translation unit per instantiation).  Phase A and Phase C ran on cc87534 (in phase_a/regress.txt "br" is that
build; phase_a/regressx.txt compares its 4:4:4 / 10-bit outputs with f8874bd).

env.txt                 machine, driver, CUDA, compilers, decoders, codec versions
formats_*.log           bench/formats.py output (speed, then quality; results.json / summary.md one level up)
phase_a/                unit tests, ctest, CPU fuzz, 8-bit regression against main, decoder gates (dav1d 1.4.1 and
                        aomdec 3.8.2 must reproduce the encoder's reconstruction byte for byte)
phase_b/timing_*        GP_ENCODERS=4, gamut set repeated to 1,434 MP, bench.gpu_speed (warm-up + 3 timed runs per
                        round), rounds interleaved with alternating order
phase_b/recon_*         GP_ENCODERS=1, gamut set: closed-loop (k_recon) wall time; "alone" = f8874bd with only the
                        8-bit 4:2:0 k_recon instantiation compiled
phase_b/ptxas_*         ptxas -v of k_recon.cu (registers, stack, spills per function)
scripts/                setup and test scripts run on the instance (bench/formats.py is in the repository)
