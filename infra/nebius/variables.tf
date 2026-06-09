variable "project_id" {
  type        = string
  description = "Nebius project ID (parent_id)"
}

variable "instance_name" {
  type    = string
  default = "cuda-lab"
}

variable "platform" {
  type    = string
  default = "gpu-l40s-a"
  # Alternatives if L40S exhausted:
  #   gpu-l40s-d  -> presets: 1gpu-16vcpu-96gb, 1gpu-32vcpu-192gb
  #   gpu-h100-sxm -> presets: 1gpu-16vcpu-200gb, 8gpu-128vcpu-1600gb
}

variable "preset" {
  type    = string
  default = "1gpu-8vcpu-32gb"
}

variable "subnet_id" {
  type    = string
  default = "vpcsubnet-e00h8svb3ekymm7r0b"
}

variable "ssh_public_key" {
  type = string
}

