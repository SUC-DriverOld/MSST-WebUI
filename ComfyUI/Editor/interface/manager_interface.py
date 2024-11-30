import platform
import psutil
import cpuinfo
import os
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class SystemInfo:
    @staticmethod
    def get_os_info():
        """获取操作系统信息"""
        os_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        # Windows特定信息
        if platform.system() == 'Windows':
            os_info['edition'] = platform.win32_edition()
            
        return os_info
    
    @staticmethod
    def get_cpu_info():
        """获取CPU详细信息"""
        cpu = cpuinfo.get_cpu_info()
        cpu_info = {
            'brand': cpu['brand_raw'],
            'architecture': cpu['arch'],
            'bits': cpu['bits'],
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'frequency_min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
            'frequency_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'cache': {
                'l1': cpu.get('l1_data_cache_size'),
                'l2': cpu.get('l2_cache_size'),
                'l3': cpu.get('l3_cache_size')
            }
        }
        return cpu_info
    
    @staticmethod
    def get_gpu_info():
        """获取GPU信息"""
        if not GPU_AVAILABLE:
            return []
            
        gpu_info = []
        try:
            for gpu in GPUtil.getGPUs():
                gpu_info.append({
                    'name': gpu.name,
                    'id': gpu.id,
                    'load': f"{gpu.load * 100:.1f}%",
                    'memory': {
                        'total': f"{gpu.memoryTotal:.0f} MB",
                        'used': f"{gpu.memoryUsed:.0f} MB",
                        'free': f"{gpu.memoryFree:.0f} MB",
                        'percentage': f"{(gpu.memoryUsed/gpu.memoryTotal)*100:.1f}%"
                    },
                    'temperature': f"{gpu.temperature} °C",
                    'uuid': gpu.uuid
                })
        except Exception as e:
            print(f"Error getting GPU info: {e}")
        return gpu_info
    
    @staticmethod
    def get_memory_info():
        """获取内存信息"""
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        memory_info = {
            'ram': {
                'total': f"{virtual_memory.total / (1024**3):.2f} GB",
                'available': f"{virtual_memory.available / (1024**3):.2f} GB",
                'used': f"{virtual_memory.used / (1024**3):.2f} GB",
                'percentage': f"{virtual_memory.percent}%"
            },
            'swap': {
                'total': f"{swap_memory.total / (1024**3):.2f} GB",
                'used': f"{swap_memory.used / (1024**3):.2f} GB",
                'free': f"{swap_memory.free / (1024**3):.2f} GB",
                'percentage': f"{swap_memory.percent}%"
            }
        }
        return memory_info
    
    @staticmethod
    def get_all_info():
        """获取所有系统信息"""
        return {
            'os': SystemInfo.get_os_info(),
            'cpu': SystemInfo.get_cpu_info(),
            'gpu': SystemInfo.get_gpu_info(),
            'memory': SystemInfo.get_memory_info()
        }

# 使用示例
if __name__ == '__main__':
    info = SystemInfo.get_all_info()
    
    # 打印系统信息
    print("=== Operating System ===")
    for key, value in info['os'].items():
        print(f"{key}: {value}")
        
    print("\n=== CPU ===")
    for key, value in info['cpu'].items():
        print(f"{key}: {value}")
        
    print("\n=== GPU ===")
    for i, gpu in enumerate(info['gpu']):
        print(f"\nGPU {i}:")
        for key, value in gpu.items():
            print(f"{key}: {value}")
            
    print("\n=== Memory ===")
    for key, value in info['memory'].items():
        print(f"\n{key.upper()}:")
        for k, v in value.items():
            print(f"{k}: {v}")