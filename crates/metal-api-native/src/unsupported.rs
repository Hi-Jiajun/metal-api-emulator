use crate::refusal;
use metal_api_core::provider::{
    CompiledComputePipeline, CompletionDisposition, CompletionToken, ComputeProvider, DeviceEpoch,
    PipelineCompileRequest, PipelineProvider, ProviderCapabilities, ProviderError,
    ProviderErrorClass, ProviderPhase, ProviderSubmission, ValidatedComputeTrace,
};
use std::time::Duration;

/// Portable API shape. Construction returns a typed capability refusal on
/// non-macOS hosts, and this uninhabited type cannot fake a provider identity.
pub enum NativeMetalProvider {}

impl NativeMetalProvider {
    pub fn new() -> Result<Self, ProviderError> {
        Err(refusal(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "native_metal_platform_unavailable",
        ))
    }

    pub fn device_name(&self) -> &str {
        match *self {}
    }
}

impl ComputeProvider for NativeMetalProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        match *self {}
    }
    fn submit(&self, _: ValidatedComputeTrace) -> Result<ProviderSubmission, ProviderError> {
        match *self {}
    }
    fn wait(
        &self,
        _: CompletionToken,
        _: Duration,
    ) -> Result<CompletionDisposition, ProviderError> {
        match *self {}
    }
}

impl PipelineProvider for NativeMetalProvider {
    fn device_epoch(&self) -> DeviceEpoch {
        match *self {}
    }
    fn compile(&self, _: PipelineCompileRequest) -> Result<CompiledComputePipeline, ProviderError> {
        match *self {}
    }
    fn release_pipeline(&self, _: &CompiledComputePipeline) -> Result<(), ProviderError> {
        match *self {}
    }
    fn release_completion(&self, _: CompletionToken) -> Result<(), ProviderError> {
        match *self {}
    }
}
