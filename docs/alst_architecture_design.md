# ALST Architecture Design

## Overview
This document outlines the architecture design for migrating LLaMA-Factory's sequence parallelism from the current monkey-patching approach to DeepSpeed's Arctic Long Sequence Training (ALST).

## Current Architecture Issues
- **Fragile monkey-patching**: Directly modifies `transformers.modeling_flash_attention_utils._flash_attention_forward`
- **External dependencies**: Requires separate `ring-flash-attn` and `yunchang` packages
- **Incomplete implementation**: "llama3" mode raises NotImplementedError
- **No DeepSpeed integration**: Current SP operates independently of DeepSpeed features

## New ALST Architecture

### Core Components
1. **DeepSpeedSequenceParallel** - New unified SP interface
2. **ALSTAttentionWrapper** - Native DeepSpeed attention integration
3. **ALSTDataAdapter** - Data processing for ALST
4. **ALSTConfig** - Configuration management
5. **CompatibilityLayer** - Backward compatibility support

### Interface Design

```python
class DeepSpeedSequenceParallel:
    """Unified interface for DeepSpeed ALST sequence parallelism."""
    
    def __init__(self, model_args: ModelArguments):
        self.model_args = model_args
        self.sp_group = None
        self.attention_wrapper = None
        
    def initialize_sp_group(self) -> ProcessGroup:
        """Initialize sequence parallel process group."""
        pass
        
    def wrap_attention_modules(self, model) -> None:
        """Replace attention modules with ALST-enabled versions.""" 
        pass
        
    def get_data_adapter(self) -> ALSTDataAdapter:
        """Get data adapter for ALST processing."""
        pass
```

### Backward Compatibility Strategy

1. **Mode Detection**: 
   - `deepspeed-alst` → Use new ALST implementation
   - `zigzag-ring`, `ulysses`, `llama3` → Use compatibility layer or legacy mode

2. **Configuration Migration**:
   - Automatic parameter mapping from old to new format
   - Warning messages for deprecated parameters
   - Migration utilities for existing configurations

3. **Gradual Migration Path**:
   - Phase 1: Add ALST support alongside existing implementation
   - Phase 2: Make ALST default for supported configurations  
   - Phase 3: Deprecate old implementations (with warnings)
   - Phase 4: Remove legacy code (major version bump)

## File Structure Changes

### New Files
- `src/llamafactory/model/model_utils/deepspeed_sequence_parallel.py` - Core ALST implementation
- `src/llamafactory/model/model_utils/alst_config.py` - ALST configuration management  
- `src/llamafactory/data/processor/alst_data_adapter.py` - ALST data processing
- `src/llamafactory/extras/alst_utils.py` - ALST utility functions
- `src/llamafactory/extras/migration_utils.py` - Configuration migration tools

### Modified Files
- `src/llamafactory/model/loader.py` - Add ALST model loading
- `src/llamafactory/train/*/trainer.py` - ALST integration in training
- `src/llamafactory/data/loader.py` - ALST data loading support

## Implementation Phases

### Phase 1: Foundation (Current)
- [x] Update dependencies to DeepSpeed 0.17.4+
- [x] Add ALST configuration parameters
- [ ] Create core ALST interface classes

### Phase 2: Core Implementation  
- [ ] Implement DeepSpeedSequenceParallel class
- [ ] Create ALSTAttentionWrapper with UlyssesSPAttentionHF
- [ ] Add model loading integration

### Phase 3: Data & Training
- [ ] Implement ALSTDataAdapter with UlyssesSPDataLoaderAdapter  
- [ ] Update training pipeline for ALST
- [ ] Add configuration validation and migration

### Phase 4: Advanced Features
- [ ] Implement sequence tiling optimizations
- [ ] Add llama3 mode using ALST
- [ ] Performance monitoring and profiling

### Phase 5: Testing & Documentation
- [ ] Comprehensive test suite
- [ ] Migration tools and documentation
- [ ] Performance benchmarks

## Benefits of New Architecture

1. **Performance**: Up to 2.5x throughput improvements, 10x communication reduction
2. **Scalability**: Support for 15M+ token sequences
3. **Reliability**: Official DeepSpeed support vs custom monkey-patching
4. **Maintainability**: Cleaner interfaces, better error handling
5. **Future-proof**: Access to latest ALST research developments

## Migration Timeline

- **Week 1-2**: Foundation and core implementation (Phases 1-2)
- **Week 3**: Data pipeline and training integration (Phase 3)  
- **Week 4**: Advanced features and optimizations (Phase 4)
- **Week 5-6**: Testing, documentation, and migration tools (Phase 5)

## Risks and Mitigation

1. **Breaking Changes**: Maintain backward compatibility layer
2. **Performance Regression**: Comprehensive benchmarking before switch
3. **Configuration Complexity**: Provide clear migration guides and tools
4. **Dependency Issues**: Careful version management and testing