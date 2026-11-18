# Distilled group communication with MAPPO

Run the complementary pursuit example through the standard trainer.

```sh
python main.py train --config=examples/distilled_group_communication_mappo.yaml
```

Set `trainer.train_device` to one learner device. The rollout block controls
parallel collection independently. See [the native environment notes](c04_environment.md)
for the MAgent2 dependency.

## Training phases

1. Train a raw-message teacher with ordinary MAPPO, or load its actor weights.
   The student inherits the teacher's policy layers and starts with a newly
   initialized codec. The teacher receives the same observation history and
   recorded group assignments as the student.
2. Collect separate teacher episodes for offline training and validation.
   With codec auxiliary supervision, fit a prediction head on frozen teacher
   group representations, then freeze the head.
3. Update only the student codec. Replace selected sends in the frozen
   teacher with the decoded student messages. The reference distribution
   always comes from the teacher receiving its original raw messages.
4. Update the student's fusion, readout and action layers on the same teacher
   episodes. Freeze the local observation/history encoder and codec.
5. Collect fresh student rollouts for MAPPO. Online distillation and auxiliary
   prediction can independently add losses to the actor objective. The
   distillation coefficient decreases linearly with PPO actor updates.

## Configuration

- `agent_group` configures message width and bits per scalar, frame extractor,
  GRU width and bias, codec encoder and decoder, attention heads and slots,
  and the local and action heads. The custom network fields accept existing
  ModelConfig specs. Actor submodules must be deterministic per sample for
  PPO replay. Group construction stays in `group_builder`.
- To replace the history GRU, set `agent_group.history_encoder` to a ModelConfig
  that accepts frame embeddings and a boolean padding mask. For the example's
  four-frame windows use
  `{model_type: SimpleResAttSeqEnc, input_dim: 64, output_dim: 64, embed_dim: 64, num_heads: 4, max_seq_len: 4, dropout: 0.0}`.
  Use the same architecture when training or loading the teacher.
- `trainer.teacher.source: train` uses the teacher's optional `trainer`,
  `critic`, `optimizer` and `lr_scheduler` overrides. Use
  `initial_actor_checkpoint` to initialize that training. Its final native
  actor is saved at `<workdir>/teacher/checkpoints/final/agent/agent.pth`.
  For a fixed teacher, use `source: checkpoint` and `actor_checkpoint` with
  a raw actor trained under the same network configuration.
- `trainer.offline` configures both offline optimizers, episode counts, batch
  size, stage lengths and sender replacement. Validation samples are held out
  and used to report initial and final losses.
- `auxiliary_head` configures a separate prediction network, query provider,
  probe updates, training optimizer and phase weights. Labels may come from
  simulator state. Only the auxiliary loss uses these labels.
- Use `group_communication_mappo` for both rollout and replay, or
  `complementary_mappo` to also record pursuit metrics. The optional
  `trainer.message_target_kl` audits hard-code and behavior replay and can
  stop repeated PPO passes when they exceed the configured policy KL.
- Standard iteration snapshots and `checkpoints/final` contain the actor
  and critic. The separately saved auxiliary weights are used only for
  training. Use a periodic or final student checkpoint to initialize online training
  with `trainer.offline.enabled: false`. Optimizers and schedules start fresh.

## Paper variants

Start from the example and change only the listed values. With the offline
block disabled, the auxiliary head is fitted during online learning.

| Variant | YAML changes |
| --- | --- |
| Main | Use the example defaults |
| Offline only | `trainer.online_distillation.enabled: false` |
| Online only | `trainer.offline.enabled: false` |
| Neither distillation stage | Disable both switches |
| No auxiliary task | `auxiliary_head.enabled: false` |
| Progressive replacement | `trainer.offline.replacement: progressive` and `replacement_steps: [200, 200, 200]` |

The 32 four-bit codes use 16 payload bytes per sender per step. The
simulator's message protocol adds a 9-byte header. Teacher training, offline
episode budgets, and PPO iteration budgets are separate YAML settings.
