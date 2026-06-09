terraform {
  required_providers {
    nebius = {
      source  = "registry.terraform.io/nebius/nebius"
      version = ">= 0.6.8"
    }
  }
}

provider "nebius" {
  # export NEBIUS_IAM_TOKEN=$(nebius iam get-access-token)
}

resource "nebius_vpc_v1_allocation" "public_ip" {
  parent_id = var.project_id
  name      = "${var.instance_name}-ip"

  ipv4_public = {
    subnet_id = var.subnet_id
  }
}

resource "nebius_compute_v1_instance" "cuda_lab" {
  parent_id = var.project_id
  name      = var.instance_name

  resources = {
    platform = var.platform
    preset   = var.preset
  }

  boot_disk = {
    attach_mode = "READ_WRITE"
    device_id   = "boot-disk"
    managed_disk = {
      name = "${var.instance_name}-boot"
      spec = {
        size_gibibytes = 80
        type           = "NETWORK_SSD"
        source_image_family = {
          image_family = "ubuntu24.04-cuda13.0"
        }
      }
    }
  }

  network_interfaces = [
    {
      name      = "eth0"
      subnet_id = var.subnet_id
      ip_address = {}
      public_ip_address = {
        allocation_id = nebius_vpc_v1_allocation.public_ip.id
        static        = true
      }
    }
  ]

  cloud_init_user_data = <<-EOT
    #cloud-config
    users:
      - name: ubuntu
        sudo: ALL=(ALL) NOPASSWD:ALL
        shell: /bin/bash
        ssh_authorized_keys:
          - ${var.ssh_public_key}
    runcmd:
      - git clone https://github.com/mesutoezdil/Systematic-CUDA-Learning.git /home/ubuntu/Systematic-CUDA-Learning
      - chown -R ubuntu:ubuntu /home/ubuntu/Systematic-CUDA-Learning
      - echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> /home/ubuntu/.bashrc
      - echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc
  EOT
}
